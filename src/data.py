"""Data layer: featurization, cross-database matching, datasets, scaffold split.

This single module covers everything between a raw SMILES string and a ready-to-
train PyG graph:

  * featurization     — SMILES -> graph with 42-dim atom / 10-dim bond features
                        (OGB / Hu et al. 2020 convention, proposal Section 3.1);
  * Morgan fingerprint — for the RF baseline and the GNN+FP fusion head;
  * matching          — join COCONUT (structures) to ChEMBL (labels) on InChIKey
                        (proposal Sections 3.4-3.5);
  * datasets          — build / save / load PyG graph lists, class pos-weights;
  * scaffold split    — MoleculeNet Bemis-Murcko split (no train/test leakage).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional, Sequence

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from tqdm import tqdm

# =============================================================================
# Featurization
# =============================================================================
# --- atom feature vocabularies (dims: 10 + 11 + 6 + 4 + 5 + 5 + 1 = 42) -------
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]   # 10
DEGREES = list(range(11))                                               # 11
CHARGES = [-2, -1, 0, 1, 2, "other"]                                    # 6
CHIRALITIES = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]                                                                       # 4
NUM_HS = list(range(5))                                                 # 5
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]                                                                       # 5

ATOM_FEATURE_DIM = 42
NUM_ATOM_TYPES = len(ATOM_TYPES)  # first 10 dims = atom-type one-hot (mask target)

# --- bond feature vocabularies (dims: 4 + 1 + 1 + 4 = 10) ---------------------
BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]                                                                       # 4
STEREOS = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOZ,
]                                                                       # 4

BOND_FEATURE_DIM = 10

# --- Morgan fingerprint (for GNN + fingerprint fusion, 2024-25 hybrid trend) --
FP_BITS = 2048
_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def morgan_fp(smiles: str) -> Optional[torch.Tensor]:
    """ECFP4-style Morgan fingerprint as a (1, FP_BITS) float tensor, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = _FP_GEN.GetFingerprintAsNumPy(mol)
    return torch.tensor(arr, dtype=torch.float).unsqueeze(0)


def one_hot(value, choices: Sequence) -> list[int]:
    """One-hot encode ``value`` over ``choices``; the last slot is the catch-all."""
    vec = [0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    vec[idx] = 1
    return vec


def atom_features(atom: rdchem.Atom) -> list[int]:
    return (
        one_hot(atom.GetSymbol(), ATOM_TYPES)
        + one_hot(atom.GetDegree(), DEGREES)
        + one_hot(atom.GetFormalCharge(), CHARGES)
        + one_hot(atom.GetChiralTag(), CHIRALITIES)
        + one_hot(atom.GetTotalNumHs(), NUM_HS)
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
    )


def bond_features(bond: rdchem.Bond) -> list[int]:
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
        + one_hot(bond.GetStereo(), STEREOS)
    )


def smiles_to_graph(
    smiles: str,
    labels: Optional[Sequence[float]] = None,
    mask: Optional[Sequence[bool]] = None,
) -> Optional[Data]:
    """Parse a SMILES string into a PyG graph.

    Returns ``None`` if RDKit cannot parse the SMILES (common for exotic
    natural-product structures in COCONUT). Each bond becomes two directed
    edges so message passing is symmetric.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index: list[list[int]] = []
    edge_attr: list[list[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    if edge_index:  # molecules with at least one bond
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:           # single-atom molecules: no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)   # (1, num_labels)
        data.mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)    # (1, num_labels)
    return data


# =============================================================================
# Cross-database matching (COCONUT structures <-> ChEMBL labels)
# =============================================================================
# COCONUT (unlabeled natural products) and ChEMBL (activity labels) are joined on
# InChIKey — a structure hash stable across toolkits, unlike canonical SMILES.
# Matched compounds form the labeled fine-tuning set; the unmatched COCONUT
# remainder is the Stage-1 pretraining corpus. Each ChEMBL cell is 1 (active),
# 0 (tested inactive), or NaN/empty (untested -> masked, deliberately not 0).


def load_coconut(path: str) -> pd.DataFrame:
    """COCONUT export with at least coconut_id, canonical_smiles, inchikey."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["canonical_smiles", "inchikey"]).drop_duplicates("inchikey")
    return df.reset_index(drop=True)


def load_chembl_labels(path: str, label_cols: list[str]) -> pd.DataFrame:
    """ChEMBL label table: inchikey + one column per activity (1/0/NaN)."""
    df = pd.read_csv(path)
    missing = [c for c in ["inchikey", *label_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"ChEMBL label file is missing columns: {missing}")
    return df.drop_duplicates("inchikey").reset_index(drop=True)


def match_datasets(
    coconut: pd.DataFrame,
    chembl: pd.DataFrame,
    label_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split COCONUT into (labeled, pretrain-only) by InChIKey membership.

    Returns:
        labeled:  COCONUT ∩ ChEMBL, carrying the SMILES + label columns.
        pretrain: COCONUT compounds with no ChEMBL match (Stage 1 only).
    """
    labeled = coconut.merge(
        chembl[["inchikey", *label_cols]], on="inchikey", how="inner"
    )
    pretrain = coconut[~coconut["inchikey"].isin(labeled["inchikey"])].copy()

    print(f"Matched (labeled) compounds : {len(labeled)}")
    print(f"Unlabeled (Stage 1 only)    : {len(pretrain)}")
    return labeled.reset_index(drop=True), pretrain.reset_index(drop=True)


# =============================================================================
# Datasets: build / save / load PyG graph lists
# =============================================================================
def build_graph_list(
    df: pd.DataFrame,
    label_cols: list[str] | None,
    smiles_col: str = "canonical_smiles",
) -> list:
    """Convert a dataframe of SMILES (+ optional labels) into PyG graphs.

    When ``label_cols`` is given, each graph carries ``y`` (labels, 0 for the
    unknown entries) and ``mask`` (True where a label is actually known). When
    it is ``None`` the graphs are unlabeled (pretraining corpus).
    """
    graphs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="featurizing"):
        if label_cols:
            raw = [row.get(c, float("nan")) for c in label_cols]
            mask = [not (isinstance(v, float) and math.isnan(v)) and pd.notna(v) for v in raw]
            labels = [int(v) if m else 0 for v, m in zip(raw, mask)]
            graph = smiles_to_graph(row[smiles_col], labels, mask)
        else:
            graph = smiles_to_graph(row[smiles_col])
        if graph is not None:
            graph.smiles = row[smiles_col]
            graphs.append(graph)
    return graphs


def save_graphs(graphs: list, path: str) -> None:
    torch.save(graphs, path)
    print(f"Saved {len(graphs)} graphs -> {path}")


def load_graphs(path: str) -> list:
    # weights_only=False: these are PyG Data objects, not bare tensors.
    return torch.load(path, weights_only=False)


def compute_pos_weight(graphs: list, num_labels: int) -> torch.Tensor:
    """Per-label inverse class frequency: (#neg known) / (#pos known).

    Used to up-weight rare positives in the masked BCE loss. Labels with no
    positives fall back to a weight of 1.0.
    """
    pos = torch.zeros(num_labels)
    neg = torch.zeros(num_labels)
    for g in graphs:
        y = g.y.squeeze(0)
        m = g.mask.squeeze(0)
        pos += (y * m).float()
        neg += ((1 - y) * m).float()
    weight = torch.where(pos > 0, neg / pos.clamp(min=1), torch.ones(num_labels))
    return weight


# =============================================================================
# Scaffold split (MoleculeNet protocol)
# =============================================================================
# Grouping molecules by Bemis-Murcko scaffold and assigning whole scaffold groups
# to a single split guarantees that structurally similar molecules never straddle
# train and test, avoiding the optimistic leakage of a random split.


def get_scaffold(smiles: str, include_chirality: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(
    smiles_list: list[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """Return train/val/test index lists.

    Largest scaffold groups are placed first so the big, common scaffolds land
    in the training set — the standard deterministic MoleculeNet ordering.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        # Molecules RDKit cannot parse get their own singleton bucket.
        scaffold_to_indices[scaffold if scaffold is not None else f"__invalid_{idx}"].append(idx)

    groups = sorted(scaffold_to_indices.values(), key=lambda g: (len(g), g[0]), reverse=True)

    n_total = len(smiles_list)
    n_train_cutoff = frac_train * n_total
    n_val_cutoff = (frac_train + frac_val) * n_total

    train, val, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= n_train_cutoff:
            train += group
        elif len(train) + len(val) + len(group) <= n_val_cutoff:
            val += group
        else:
            test += group
    return train, val, test


def assert_no_scaffold_overlap(smiles_list, train_idx, test_idx) -> None:
    """Sanity check used at split time; raises if any scaffold is shared."""
    train_scaffolds = {get_scaffold(smiles_list[i]) for i in train_idx}
    test_scaffolds = {get_scaffold(smiles_list[i]) for i in test_idx}
    overlap = (train_scaffolds & test_scaffolds) - {None}
    assert not overlap, f"Scaffold overlap between train and test: {len(overlap)} scaffolds"
