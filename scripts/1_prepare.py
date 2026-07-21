"""Stage 1 — Data preparation.

Subcommands:
  synthetic       Write tiny synthetic COCONUT/ChEMBL CSVs to smoke-test the pipeline.
  extract-chembl  Extract antiparasitic activity labels from a local ChEMBL SQLite dump.
  build-graphs    Match COCONUT<->ChEMBL, featurize, scaffold-split -> data/processed/.
  build-shards    Featurize the FULL unlabeled COCONUT corpus into streamable shards.

Examples:
    python scripts/1_prepare.py synthetic --n 400
    python scripts/1_prepare.py extract-chembl --chembl-db /path/to/chembl_37.db
    python scripts/1_prepare.py build-graphs
    python scripts/1_prepare.py build-shards --shard-size 120000
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import gc
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from src.data import (
    assert_no_scaffold_overlap,
    build_graph_list,
    load_chembl_labels,
    load_coconut,
    match_datasets,
    save_graphs,
    scaffold_split,
)
from src.train import add_common_args, load_config, set_seed

csv.field_size_limit(sys.maxsize)

# label -> SQL LIKE pattern matched against assays.assay_organism
PATHOGENS = {
    "antimalarial": "%Plasmodium%",
    "antitrypanosomal": "%Trypanosoma%",
    "antileishmanial": "%Leishmania%",
    "antitubercular": "%Mycobacterium tuberculosis%",
}
LABELS = list(PATHOGENS)
ACTIVE_PCHEMBL = 5.0  # pchembl >= 5  <=>  IC50/EC50 <= 10 uM


# =============================================================================
# synthetic — tiny CSVs to exercise the pipeline before the real download
# =============================================================================
# A pool of small, valid drug-like / natural-product SMILES.
SEED_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccccc1",
    "CCO", "CC(=O)O", "C1CCCCC1", "c1ccncc1", "OC(=O)c1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "C1=CC=C2C(=C1)C=CC=C2", "CC1=CC(=O)CC(C)(C)C1", "OCC1OC(O)C(O)C(O)C1O",
    "c1cc2ccc3cccc4ccc(c1)c2c34", "CC(=O)Nc1ccc(O)cc1", "Cc1ccccc1",
    "NCCc1ccc(O)c(O)c1", "C(C(=O)O)N", "OC1=CC(=O)c2ccccc2O1",
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CC(N)C(=O)O", "c1ccc(cc1)c1ccccc1",
    "O=C(O)CCc1ccccc1", "Cc1ccc(O)cc1", "CCCCCCCC(=O)O",
    "CC1=C(C(=O)c2ccccc2C1=O)C", "Oc1ccccc1O", "Nc1ccccc1",
    "CC(C)(C)c1ccccc1O", "C1=CC2=CC=CC=C2C=C1",
]
FRAGS = ["", "C", "CC", "CCC", "O", "N", "Cl", "Br", "F", "C(=O)O", "OC", "CO",
         "CN", "CCO", "CCl", "C(C)C", "OCC", "NC"]


def _variant(base: str, frag: str) -> str | None:
    """Append a fragment to a base SMILES; return canonical form or None."""
    mol = Chem.MolFromSmiles(base + frag)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _unique_pool() -> dict:
    """Deterministically enumerate base x fragment combos into {inchikey: smiles}."""
    pool: dict[str, str] = {}
    for base in SEED_SMILES:
        for frag in FRAGS:
            smi = _variant(base, frag)
            if smi is None:
                continue
            ikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
            pool.setdefault(ikey, smi)
    return pool


def cmd_synthetic(args):
    cfg = load_config(args.config)
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    label_cols = list(cfg.labels)

    # Build a unique compound set keyed by InChIKey from the enumerated pool.
    pool = _unique_pool()
    keys = list(pool.keys())
    rng.shuffle(keys)
    n = min(args.n, len(keys))
    if n < args.n:
        print(f"(pool exhausted: using {n} unique compounds instead of {args.n})")
    inchikeys = keys[:n]
    smiles = [pool[k] for k in inchikeys]

    coconut = pd.DataFrame({
        "coconut_id": [f"CNP{i:06d}" for i in range(len(inchikeys))],
        "canonical_smiles": smiles,
        "inchikey": inchikeys,
        "name": [f"compound_{i}" for i in range(len(inchikeys))],
    })

    # ChEMBL: a random subset, with sparse multi-label data (0/1/NaN).
    n_match = int(len(inchikeys) * args.match_frac)
    match_idx = rng.sample(range(len(inchikeys)), n_match)
    chembl_rows = []
    for i in match_idx:
        row = {"inchikey": inchikeys[i]}
        for c in label_cols:
            r = rng.random()
            row[c] = np.nan if r < 0.5 else (1 if r < 0.65 else 0)   # ~50% untested
        chembl_rows.append(row)
    chembl = pd.DataFrame(chembl_rows)

    raw = Path(cfg.paths.processed_dir).parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    coconut.to_csv(raw / "coconut.csv", index=False)
    chembl.to_csv(raw / "chembl_labels.csv", index=False)
    print(f"Wrote {len(coconut)} COCONUT rows and {len(chembl)} ChEMBL label rows to {raw}")


# =============================================================================
# extract-chembl — build data/raw/chembl_labels.csv from a ChEMBL SQLite dump
# =============================================================================
# Each label is sourced from *phenotypic* assays keyed by assays.assay_organism
# (not the small in-vivo assay_classification table, too sparse for natural
# products). Per compound, per label: 1 = active (pchembl >= 5, or comment
# 'active'); 0 = tested but inactive; empty = never tested -> masked downstream.
# Output is restricted to compounds present in COCONUT (matched by InChIKey).


def _label_keys(conn: sqlite3.Connection) -> tuple[dict[str, set], dict[str, set]]:
    """Single-pass extraction of tested/active InChIKeys per label.

    First materialise the (small) set of relevant assays with their label into a
    temp table, then make ONE aggregation pass over activities -> structures via
    the indexed assay_id / molregno columns — far cheaper than one join per label.
    """
    case = "\n".join(
        f"WHEN assay_organism LIKE '{pat}' THEN '{label}'"
        for label, pat in PATHOGENS.items()
    )
    where = " OR ".join(f"assay_organism LIKE '{pat}'" for pat in PATHOGENS.values())
    conn.executescript(
        f"""
        DROP TABLE IF EXISTS assay_label;
        CREATE TEMP TABLE assay_label AS
            SELECT assay_id, CASE {case} END AS lab
            FROM assays WHERE {where};
        CREATE INDEX tmp_al_assay ON assay_label(assay_id);
        """
    )

    sql = f"""
        SELECT cs.standard_inchi_key AS k, al.lab AS lab,
               MAX(CASE WHEN (act.pchembl_value IS NOT NULL AND act.pchembl_value >= {ACTIVE_PCHEMBL})
                             OR act.activity_comment IN ('active','Active')
                        THEN 1 ELSE 0 END) AS active
        FROM assay_label al
        JOIN activities act ON act.assay_id = al.assay_id
        JOIN compound_structures cs ON cs.molregno = act.molregno
        GROUP BY cs.standard_inchi_key, al.lab
    """
    tested = {label: set() for label in LABELS}
    active = {label: set() for label in LABELS}
    for key, lab, is_active in conn.execute(sql):
        if not key:
            continue
        tested[lab].add(key)
        if is_active:
            active[lab].add(key)
    return tested, active


def cmd_extract_chembl(args):
    # COCONUT InChIKeys: only matched compounds become labeled rows.
    with open(args.coconut, newline="") as f:
        coconut_keys = {row["inchikey"].strip() for row in csv.DictReader(f) if row["inchikey"]}
    print(f"COCONUT InChIKeys: {len(coconut_keys):,}")

    conn = sqlite3.connect(args.chembl_db)
    tested_all, active_all = _label_keys(conn)
    conn.close()

    tested = {label: tested_all[label] & coconut_keys for label in LABELS}
    active = {label: active_all[label] & coconut_keys for label in LABELS}
    for label in LABELS:
        n_pos = len(active[label])
        n_neg = len(tested[label]) - n_pos
        print(f"{label:<18} tested={len(tested[label]):>6,}  active={n_pos:>5,}  inactive={n_neg:>6,}")

    # Every compound tested for at least one pathogen becomes a labeled row.
    all_keys = set().union(*tested.values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inchikey", *LABELS])
        for key in sorted(all_keys):
            cells = [key]
            for label in LABELS:
                if key in active[label]:
                    cells.append(1)
                elif key in tested[label]:
                    cells.append(0)
                else:
                    cells.append("")  # untested -> masked
            w.writerow(cells)
    print(f"Wrote {len(all_keys):,} labeled compounds -> {args.out}")


# =============================================================================
# build-graphs — match, scaffold-split, featurize the labeled + pretrain sets
# =============================================================================
def cmd_build_graphs(args):
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)

    out = Path(cfg.paths.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_cols = list(cfg.labels)

    # 1. Load + match -------------------------------------------------------
    coconut = load_coconut(cfg.paths.coconut_csv)
    chembl = load_chembl_labels(cfg.paths.chembl_csv, label_cols)
    labeled_df, pretrain_df = match_datasets(coconut, chembl, label_cols)

    # 2. Scaffold split of the labeled set ----------------------------------
    smiles = labeled_df["canonical_smiles"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(
        smiles, cfg.split.frac_train, cfg.split.frac_val, cfg.split.frac_test
    )
    assert_no_scaffold_overlap(smiles, train_idx, test_idx)
    print(f"Split sizes  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # 3. Featurize + save ---------------------------------------------------
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        graphs = build_graph_list(labeled_df.iloc[idx], label_cols)
        save_graphs(graphs, out / f"labeled_{name}.pt")

    if args.max_pretrain and len(pretrain_df) > args.max_pretrain:
        pretrain_df = pretrain_df.sample(n=args.max_pretrain, random_state=args.seed or cfg.seed)
        print(f"Subsampled pretrain corpus to {len(pretrain_df)} compounds")
    pretrain_graphs = build_graph_list(pretrain_df, label_cols=None)
    save_graphs(pretrain_graphs, out / "pretrain.pt")
    print("Done.")


# =============================================================================
# build-shards — featurize the FULL unlabeled COCONUT corpus into shards
# =============================================================================
# The default build-graphs subsamples the unlabeled corpus to 200k. Here we
# featurize *all* ~714k unmatched COCONUT compounds, written as shards so
# Stage-1 pretraining can stream them one shard at a time (the full set does not
# fit in 16 GB RAM at once).


def cmd_build_shards(args):
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)
    out = Path(cfg.paths.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_cols = list(cfg.labels)

    # Same match as build-graphs: pretrain corpus is COCONUT minus the labeled set.
    coconut = load_coconut(cfg.paths.coconut_csv)
    chembl = load_chembl_labels(cfg.paths.chembl_csv, label_cols)
    _, pretrain_df = match_datasets(coconut, chembl, label_cols)
    n = len(pretrain_df)
    print(f"Full unlabeled pretrain corpus: {n:,} compounds")

    # Clear any stale shards from a previous, differently-sized run.
    for old in out.glob("pretrain_shard_*.pt"):
        old.unlink()

    shard_size = args.shard_size
    n_shards = (n + shard_size - 1) // shard_size
    kept = 0
    for s in range(n_shards):
        chunk = pretrain_df.iloc[s * shard_size:(s + 1) * shard_size]
        graphs = build_graph_list(chunk, label_cols=None)   # unlabeled
        path = out / f"pretrain_shard_{s:03d}.pt"
        save_graphs(graphs, path)
        kept += len(graphs)
        del graphs
        gc.collect()
    print(f"Done. {kept:,} graphs across {n_shards} shards -> {out}/pretrain_shard_*.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synthetic", help="tiny synthetic CSVs for smoke-testing")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=400, help="Number of COCONUT rows.")
    p.add_argument("--match-frac", type=float, default=0.4,
                   help="Fraction of COCONUT compounds that also appear in ChEMBL.")
    p.set_defaults(func=cmd_synthetic)

    p = sub.add_parser("extract-chembl", help="ChEMBL SQLite -> chembl_labels.csv")
    p.add_argument("--chembl-db", required=True, help="path to chembl_*.db (SQLite)")
    p.add_argument("--coconut", default="data/raw/coconut.csv")
    p.add_argument("--out", default="data/raw/chembl_labels.csv")
    p.set_defaults(func=cmd_extract_chembl)

    p = add_common_args(sub.add_parser("build-graphs", help="featurize + scaffold split"))
    p.add_argument("--max-pretrain", type=int, default=200_000,
                   help="Subsample the unlabeled pretrain corpus to this many "
                        "compounds (0 = use all).")
    p.set_defaults(func=cmd_build_graphs)

    p = add_common_args(sub.add_parser("build-shards", help="full-corpus pretrain shards"))
    p.add_argument("--shard-size", type=int, default=120_000)
    p.set_defaults(func=cmd_build_shards)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
