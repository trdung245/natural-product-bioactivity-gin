# Data

The datasets are **not** committed (they are large). Download them here before
running the pipeline. The `raw/` and `processed/` folders are kept in git only via
`.gitkeep` placeholders.

## Expected layout

```
data/
├── raw/
│   ├── coconut.csv         # natural products (COCONUT)
│   └── chembl_labels.csv   # activity labels (ChEMBL, one row per InChIKey)
└── processed/              # written by `scripts/1_prepare.py build-graphs`
    ├── labeled_train.pt
    ├── labeled_val.pt
    ├── labeled_test.pt
    └── pretrain.pt
```

## 1. `raw/coconut.csv` — natural-product structures

Download the **CSV (lite)** bulk export from COCONUT:
<https://coconut.naturalproducts.net/download>  →  `coconut_csv-*.csv` (csv_lite).

The lite export ships with columns `identifier` and `standard_inchi_key`; rename
them to the schema the pipeline expects:

| column             | source column        | description                        |
|--------------------|----------------------|------------------------------------|
| `coconut_id`       | `identifier`         | COCONUT identifier                 |
| `canonical_smiles` | `canonical_smiles`   | SMILES string (parsed by RDKit)    |
| `inchikey`         | `standard_inchi_key` | InChIKey — cross-database match key |
| `name`             | `name` (optional)    | compound name                      |

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("coconut_csv-XX-XXXX.csv")   # the file you downloaded
df = df.rename(columns={"identifier": "coconut_id",
                        "standard_inchi_key": "inchikey"})
df.to_csv("data/raw/coconut.csv", index=False)
PY
```

## 2. `raw/chembl_labels.csv` — activity labels

Generated from a local ChEMBL SQLite dump. Download and unpack ChEMBL 37:
<https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/>  →
`chembl_37_sqlite.tar.gz` (~4 GB compressed, ~30 GB unpacked).

```bash
tar -xzf chembl_37_sqlite.tar.gz          # -> chembl_37/chembl_37_sqlite/chembl_37.db
```

Four antiparasitic activities, each sourced from *phenotypic* assays keyed by
`assays.assay_organism` (the in-vivo `assay_classification` table is too sparse
for natural products). One row per compound, keyed by InChIKey, one column per
activity. Each cell is:

- `1` — active (pchembl ≥ 5, i.e. ≤ 10 µM, or flagged active),
- `0` — tested inactive (screened against that pathogen, no active record),
- empty / `NaN` — **untested → masked** (not treated as inactive).

Required columns:

```
inchikey, antimalarial, antitrypanosomal, antileishmanial, antitubercular
```

Build it (after `coconut.csv` exists), matched to the COCONUT compounds:

```bash
python scripts/1_prepare.py extract-chembl \
    --chembl-db chembl_37/chembl_37_sqlite/chembl_37.db \
    --coconut data/raw/coconut.csv \
    --out data/raw/chembl_labels.csv
```

## Don't have the data yet?

`scripts/1_prepare.py synthetic` writes tiny synthetic `coconut.csv` /
`chembl_labels.csv` files (real SMILES + InChIKeys) so you can exercise the full
pipeline before the real downloads finish. The numbers it produces are
meaningless — it only validates that the code runs.

```bash
python scripts/1_prepare.py synthetic --n 400
```
