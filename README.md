# Multi-Label Bioactivity Prediction of Natural Products with a GIN

Domain-adaptive pretraining of a **Graph Isomorphism Network (GIN)** for
multi-label antiparasitic bioactivity prediction of natural products.

## Abstract

Natural products are the historical wellspring of anti-infective drugs, but the
molecules with measured activity are a tiny, sparsely labeled fraction of known
chemical space. We ask whether **self-supervised pretraining on unlabeled natural
products (COCONUT)** yields a molecular representation that transfers to
data-scarce activity prediction. A 5-layer GIN encoder is first pretrained on
~200K–714K unlabeled COCONUT structures (attribute masking, GraphMAE, or MolCLR
contrastive), then fine-tuned to predict four organism-defined activities: 
**antimalarial, antitrypanosomal, antileishmanial, antitubercular** from
structure alone. Labels come from ChEMBL *phenotypic* screens keyed by assay
organism, giving real actives **and** tested-inactives; compounds never tested
for a given pathogen are **masked** in the loss rather than assumed inactive.
Evaluation uses a scaffold split (no train/test structural leakage) and reports
macro AUROC / AUPR / F1 against a Morgan-fingerprint + Random-Forest baseline.
Domain pretraining gives a real, multi-seed-confirmed improvement over training
from scratch; the strongest generative/contrastive encoders match the RF
baseline; and a **GIN + RF ensemble** clears it on every metric.

```
COCONUT graphs Stage 1: self-supervised pretraining──▶ domain-adapted GIN encoder
                                                                 │ weight transfer
matched COCONUT∩ChEMBL labels Stage 2: supervised fine-tune──▶ 4× sigmoid (masked BCE)
```

### Two strategies compared

| Strategy | Pretraining | Fine-tuning |
|----------|-------------|-------------|
| **A** (baseline) | none — random init | ChEMBL labels only |
| **B** (proposed) | self-supervised on COCONUT graphs | matched ChEMBL labels |

Plus a Morgan-fingerprint + Random-Forest baseline, a label-handling ablation
(masked vs. naive loss), and several extensions (§ *Results*).

## Install

PyTorch Geometric must match your PyTorch/CUDA build exactly. Install Torch
first, then the matching PyG wheel, then the rest:

```bash
pip install torch                       # pick the build for your CUDA/CPU/MPS
pip install torch_geometric
pip install -r requirements.txt         # rdkit, scikit-learn, pandas, matplotlib, ...
```

Quick check:

```python
from rdkit import Chem
from torch_geometric.nn import GINEConv
assert Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O") is not None
```

## Get the data

See [`data/README.md`](data/README.md) for formats and download links:

- **COCONUT** natural products: CSV (lite) from
  <https://coconut.naturalproducts.net/download>
- **ChEMBL 37** SQLite dump: `chembl_37_sqlite.tar.gz` from
  <https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/>

To smoke-test the whole pipeline **without** the real data:

```bash
python scripts/1_prepare.py synthetic --n 400
```

## Run the pipeline

Every command reads knobs from [`configs/default.yaml`](configs/default.yaml);
override on the command line (e.g. `--seed`, `--device`, `--epochs`).

```bash
# 1) Prepare labels, featurization, scaffold split
python scripts/1_prepare.py extract-chembl --chembl-db /path/to/chembl_37.db
python scripts/1_prepare.py build-graphs

# 2) Pretrain the encoder (Stage 1 — pick one pretext)
python scripts/2_pretrain.py mask                       # attribute masking (200k)
#   optional stronger pretexts on the full 714k corpus:
python scripts/1_prepare.py build-shards
python scripts/2_pretrain.py graphmae   
python scripts/2_pretrain.py contrastive

# 3) Fine-tune (Stage 2)
python scripts/3_finetune.py --strategy A               # baseline (random init)
python scripts/3_finetune.py --strategy B               # proposed (attribute-mask encoder)
python scripts/3_finetune.py --strategy B --naive-loss  # label-handling ablation
python scripts/3_finetune.py --strategy B --encoder checkpoints/gin_graphmae.pt --tag strategy_b_gmae

# 4) Baselines & evaluation
python scripts/4_evaluate.py baseline-rf                # Morgan FP + Random Forest
python scripts/4_evaluate.py report --model strategy_b  # comparison table + tau sweep
python scripts/4_evaluate.py calibrate                  # per-label threshold tuning

# 5) Extra experiments
python scripts/experiments.py multiseed --seeds 42 1 7  # variance vs. real gaps
python scripts/experiments.py ensemble --gin strategy_b_gmae
python scripts/experiments.py explain  --model strategy_b_gmae
```

Each grouped script is a small CLI run `python scripts/<name>.py -h` (or
`<name>.py <subcommand> -h`) for its options.

## Layout

```
natural-product-bioactivity-gin/
├── configs/default.yaml     # all hyperparameters
├── src/
│   ├── data.py              # featurization, InChIKey matching, datasets, scaffold split
│   ├── model.py             # GIN encoder + heads (classifier/mask/GraphMAE/MolCLR) + losses
│   ├── train.py             # config loader, seeding/device, shared fine-tuning loop
│   └── metrics.py           # AUROC / AUPR / macro-F1 (mask-aware)
├── scripts/
│   ├── 1_prepare.py         # synthetic | extract-chembl | build-graphs | build-shards
│   ├── 2_pretrain.py        # mask | graphmae | contrastive
│   ├── 3_finetune.py        # Strategy A / B supervised fine-tuning
│   ├── 4_evaluate.py        # baseline-rf | report | calibrate
│   └── experiments.py       # multiseed | *-decoupled | ensemble | explain
├── data/                    # raw + processed (gitignored; see data/README.md)
├── checkpoints/             # saved encoder / classifier weights (gitignored)
└── results/                 # metrics JSON/CSV + figures (gitignored)
```

The 16 original pipeline scripts and 13 `src` modules were consolidated into
these 5 scripts and 4 modules for readability — each script groups one pipeline
stage as argparse subcommands, and each module groups one concern.

## Results

Dataset: **14,352** labeled COCONUT∩ChEMBL compounds over four labels, scaffold-
split **11,481 / 1,435 / 1,436** (train/val/test). Test-set **macro** metrics.

### Headline comparison (single seed)

| Model | AUROC | AUPR | F1 |
|-------|:-----:|:----:|:--:|
| Strategy A - GIN, no pretrain | 0.715 | 0.290 | 0.368 |
| Strategy B - attribute-mask pretrain (200K) | 0.758 | 0.385 | 0.385 |
| Strategy B - GraphMAE pretrain (714K) | **0.777** | 0.402 | 0.396 |
| Strategy B - MolCLR pretrain (714K) | 0.771 | **0.416** | 0.383 |
| Morgan FP + Random Forest | 0.779 | 0.401 | 0.188 |
| **GIN (GraphMAE) + RF ensemble** | **0.795** | **0.423** | **0.418** |

### Findings

- **Domain pretraining helps, and it's real (not seed noise).** Across seeds
  42/1/7, Strategy B beats Strategy A on AUROC (**0.763 ± 0.008** vs 0.703 ± 0.021)
  and AUPR (0.385 vs 0.309). The largest gains are on the scarcest label
  (antileishmanial).
- **Mask, don't zero, the missing labels.** Masked BCE beats the naive
  "untested = inactive" loss on F1 (0.385 vs 0.222). Much of that gap is a
  calibration artifact. Per-label threshold tuning lifts the naive model's F1
  to 0.393.
- **A pure GIN can reach the classical baseline.** With GraphMAE (generative) or
  MolCLR (contrastive) pretraining on the full 714K corpus, Strategy B matches
  the strong Morgan-FP + RF baseline on AUROC and AUPR.
  Generative vs. contrastive is a trade: GraphMAE wins
  AUROC, MolCLR wins AUPR.
- **The GIN + RF ensemble is the strongest model.** A plain probability average
  of the GraphMAE GIN and the RF beats *both* parents on all three metrics
  (AUROC +0.016, AUPR +0.022 over RF); a per-label logistic stack matches it. The
  two model families are complementary.
- **Negative result: more data ≠ better under distribution shift.** Training on
  the broad ChEMBL antiparasitic corpus (60K+ compounds) but testing on natural
  products *hurt* every model versus training on the smaller on-distribution
  COCONUT∩ChEMBL set.

*Single-seed rows are indicative; confirmed gaps (Strategy B > A) are multi-seed.
Reproduce the table with `scripts/4_evaluate.py report` after fine-tuning, and the
robustness numbers with `scripts/experiments.py multiseed`.*

### Explainability

`scripts/experiments.py explain` attributes a prediction back to atoms. For a
sum-pooled GIN the graph embedding is `h_G = Σ_v z_v`, so each atom's
contribution to a label's pre-sigmoid logit is exactly `s_v = (∂logit/∂h_G) · z_v`
A first-order/CAM-style decomposition. It renders a 2D heat-map, a 3D conformer
PNG, and an interactive rotatable 3D view (3Dmol.js) per demo molecule.

## Design notes

- **Graph features** follow the OGB / Hu et al. (2020) convention: 42-dim atom
  vectors, 10-dim bond vectors.
- **GIN** is a 5-layer, hidden-300 network using PyG `GINEConv`, folding bond
  features into message passing via a per-layer edge projection, sum pooling for
  readout.
- **Missing labels are masked, not zeroed**: a compound untested for an activity
  contributes nothing to that label's loss (`--naive-loss` flips this to measure
  its impact).
- **Class imbalance** is handled by per-label inverse-frequency `pos_weight`.
- **Scaffold split** (MoleculeNet protocol) keeps structurally similar molecules
  out of both train and test.
- **Deployment uncertainty flag:** at inference, a compound whose maximum
  predicted probability is below a validation-tuned threshold τ is flagged
  *uncertain* rather than predicted all-inactive.

## Reproducibility

Every script seeds python/numpy/torch from `configs/default.yaml` (`seed: 42`).
Pass `--seed` to override. Pretraining is not re-run per seed (standard protocol);
only supervised fine-tuning is reseeded.
