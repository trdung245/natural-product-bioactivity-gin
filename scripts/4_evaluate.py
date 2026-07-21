"""Stage 4 — Baselines, evaluation, and calibration.

Subcommands:
  baseline-rf  Morgan fingerprint + Random Forest baseline (one RF per label).
  report       Comparison table across all models + confidence-threshold (tau) sweep.
  calibrate    Per-label F1-optimal decision thresholds (tuned on val, applied to test).

Examples:
    python scripts/4_evaluate.py baseline-rf
    python scripts/4_evaluate.py report --model strategy_b
    python scripts/4_evaluate.py calibrate --device cpu
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.data import load_graphs
from src.metrics import collect_predictions, macro_f1
from src.train import (
    add_common_args,
    build_classifier,
    build_loaders,
    get_device,
    load_config,
    set_seed,
)


# =============================================================================
# baseline-rf — Morgan fingerprint + Random Forest, one RF per label
# =============================================================================
def _rf_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.int8)


def _graphs_to_xy(graphs):
    """Return fingerprint matrix X and arrays y, mask aligned to graph order."""
    X, Y, M = [], [], []
    for g in graphs:
        fp = _rf_morgan_fp(g.smiles)
        if fp is None:
            continue
        X.append(fp)
        Y.append(g.y.squeeze(0).numpy())
        M.append(g.mask.squeeze(0).numpy())
    return np.stack(X), np.stack(Y), np.stack(M)


def cmd_baseline_rf(args):
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)

    proc = Path(cfg.paths.processed_dir)
    label_cols = list(cfg.labels)

    train = load_graphs(proc / "labeled_train.pt")
    test = load_graphs(proc / "labeled_test.pt")
    X_train, y_train, m_train = _graphs_to_xy(train)
    X_test, y_test, m_test = _graphs_to_xy(test)

    rows = {}
    for j, name in enumerate(label_cols):
        tr = m_train[:, j].astype(bool)
        te = m_test[:, j].astype(bool)
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y_train[tr, j])) < 2:
            print(f"{name}: insufficient data, skipping")
            continue

        clf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=args.seed or cfg.seed
        )
        clf.fit(X_train[tr], y_train[tr, j])
        prob = clf.predict_proba(X_test[te])[:, 1]

        y_true = y_test[te, j]
        single = len(np.unique(y_true)) < 2
        rows[name] = {
            "AUROC": float("nan") if single else roc_auc_score(y_true, prob),
            "AUPR": average_precision_score(y_true, prob),
            "F1": f1_score(y_true, (prob >= cfg.eval.threshold).astype(int), zero_division=0),
        }
        print(f"{name}: AUROC={rows[name]['AUROC']:.3f}  "
              f"AUPR={rows[name]['AUPR']:.3f}  F1={rows[name]['F1']:.3f}")

    out = Path(cfg.paths.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).T.to_csv(out / "baseline_rf.csv")
    print(f"Wrote {out / 'baseline_rf.csv'}")


# =============================================================================
# report — comparison table + tau sweep + deployment uncertainty flag
# =============================================================================
def _build_comparison_table(results_dir: Path) -> pd.DataFrame:
    """Collect every *_test.csv (+ baseline_rf.csv) into one table."""
    rows = {}
    for csv in sorted(results_dir.glob("*_test.csv")):
        name = csv.stem.replace("_test", "")
        rows[name] = pd.read_csv(csv).iloc[0].to_dict()

    rf = results_dir / "baseline_rf.csv"
    if rf.exists():
        df = pd.read_csv(rf, index_col=0)   # rows = labels, cols = AUROC/AUPR/F1
        flat = {}
        for label in df.index:
            for metric in df.columns:
                flat[f"{label}_{metric}"] = df.loc[label, metric]
        flat["macro_AUROC"] = df["AUROC"].mean()
        flat["macro_AUPR"] = df["AUPR"].mean()
        flat["macro_F1"] = df["F1"].mean()
        rows["morgan_rf"] = flat

    return pd.DataFrame(rows).T


def _tau_sweep(model, val_loader, label_names, device, tau_grid):
    """Macro-F1 across the confidence-threshold grid (proposal Section 9.3)."""
    preds, labels, masks = collect_predictions(model, val_loader, device)
    return {tau: macro_f1(preds, labels, masks, label_names, threshold=tau) for tau in tau_grid}


def _deployment_flags(model, loader, device, tau):
    """flag = 'uncertain' if max predicted prob < tau, else the active labels."""
    preds, _, _ = collect_predictions(model, loader, device)
    n_uncertain = int((preds.max(axis=1) < tau).sum())
    return n_uncertain, len(preds)


def cmd_report(args):
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)
    device = get_device(args.device)
    label_names = list(cfg.labels)

    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparison table ---------------------------------------------------
    table = _build_comparison_table(res_dir)
    table.to_csv(res_dir / "comparison_table.csv")
    print("=== Comparison table ===")
    cols = [c for c in ["macro_AUROC", "macro_AUPR", "macro_F1"] if c in table.columns]
    print(table[cols].round(3) if cols else table.round(3))

    # 2. Confidence-threshold sweep + deployment flag -----------------------
    ckpt = Path(cfg.paths.checkpoint_dir) / f"{args.model}.pt"
    if not ckpt.exists():
        print(f"\n(Checkpoint {ckpt} not found — skipping tau sweep.)")
        return

    model = build_classifier(cfg, pretrained_encoder=None, device=device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    proc = Path(cfg.paths.processed_dir)
    val_loader = DataLoader(load_graphs(proc / "labeled_val.pt"), batch_size=cfg.finetune.batch_size)
    test_loader = DataLoader(load_graphs(proc / "labeled_test.pt"), batch_size=cfg.finetune.batch_size)

    sweep = _tau_sweep(model, val_loader, label_names, device, list(cfg.eval.tau_grid))
    best_tau = max(sweep, key=sweep.get)
    print("\n=== tau sweep (val macro-F1) ===")
    for tau, f1 in sweep.items():
        print(f"  tau={tau}: macro-F1={f1:.3f}")
    print(f"Best tau on val: {best_tau} (macro-F1={sweep[best_tau]:.3f})")

    n_unc, n_tot = _deployment_flags(model, test_loader, device, best_tau)
    print(f"Deployment flag @ tau={best_tau}: "
          f"{n_unc}/{n_tot} test compounds flagged uncertain")

    with open(res_dir / "tau_sweep.json", "w") as f:
        json.dump({"sweep": sweep, "best_tau": best_tau,
                   "flagged_uncertain": n_unc, "total": n_tot}, f, indent=2)


# =============================================================================
# calibrate — per-label F1-optimal thresholds (val -> test)
# =============================================================================
# checkpoint stem -> uses fingerprint fusion?
GIN_MODELS = {
    "strategy_a": False,
    "strategy_b": False,
    "strategy_b_naive": False,
    "strategy_a_fp": True,
    "strategy_b_fp": True,
    "strategy_b_gmae": False,
    "strategy_b_fp_gmae": True,
}
GRID = np.arange(0.05, 0.96, 0.01)


def _best_thresholds(val_preds, val_labels, val_masks, num_labels):
    """F1-optimal threshold per label on the validation split."""
    thr = np.full(num_labels, 0.5)
    for j in range(num_labels):
        m = val_masks[:, j].astype(bool)
        if m.sum() == 0 or len(np.unique(val_labels[m, j])) < 2:
            continue
        yt, yp = val_labels[m, j], val_preds[m, j]
        f1s = [f1_score(yt, (yp >= t).astype(int), zero_division=0) for t in GRID]
        thr[j] = float(GRID[int(np.argmax(f1s))])
    return thr


def _per_label_f1(preds, labels, masks, thresholds, num_labels):
    f1s = []
    for j in range(num_labels):
        m = masks[:, j].astype(bool)
        if m.sum() == 0:
            f1s.append(float("nan"))
            continue
        yt = labels[m, j]
        yp = (preds[m, j] >= thresholds[j]).astype(int)
        f1s.append(f1_score(yt, yp, zero_division=0))
    return f1s


def cmd_calibrate(args):
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)
    device = get_device(args.device)
    label_names = list(cfg.labels)
    n = len(label_names)

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    res_dir = Path(cfg.paths.results_dir)
    out = {}

    for stem, use_fp in GIN_MODELS.items():
        ckpt = ckpt_dir / f"{stem}.pt"
        if not ckpt.exists():
            print(f"(skip {stem}: no checkpoint)")
            continue

        _, _, val_loader, test_loader = build_loaders(
            cfg.paths.processed_dir, cfg.finetune.batch_size, use_fp=use_fp
        )
        model = build_classifier(cfg, None, device, use_fp=use_fp)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        vp, vl, vm = collect_predictions(model, val_loader, device)
        tp, tl, tm = collect_predictions(model, test_loader, device)

        thr = _best_thresholds(vp, vl, vm, n)
        f1_fixed = _per_label_f1(tp, tl, tm, np.full(n, 0.5), n)
        f1_tuned = _per_label_f1(tp, tl, tm, thr, n)

        macro_fixed = float(np.nanmean(f1_fixed))
        macro_tuned = float(np.nanmean(f1_tuned))
        out[stem] = {
            "thresholds": {label_names[j]: round(float(thr[j]), 2) for j in range(n)},
            "f1_fixed_0.5": {label_names[j]: round(f1_fixed[j], 4) for j in range(n)},
            "f1_tuned": {label_names[j]: round(f1_tuned[j], 4) for j in range(n)},
            "macro_f1_fixed_0.5": round(macro_fixed, 4),
            "macro_f1_tuned": round(macro_tuned, 4),
            "delta": round(macro_tuned - macro_fixed, 4),
        }
        print(f"{stem:<18} macro-F1  0.5={macro_fixed:.3f}  ->  tuned={macro_tuned:.3f}  "
              f"(Δ {macro_tuned - macro_fixed:+.3f})")

    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "threshold_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {res_dir / 'threshold_calibration.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = add_common_args(sub.add_parser("baseline-rf", help="Morgan FP + Random Forest"))
    p.set_defaults(func=cmd_baseline_rf)

    p = add_common_args(sub.add_parser("report", help="comparison table + tau sweep"))
    p.add_argument("--model", default="strategy_b",
                   help="Checkpoint stem in checkpoints/ to use for the tau sweep.")
    p.set_defaults(func=cmd_report)

    p = add_common_args(sub.add_parser("calibrate", help="per-label threshold calibration"))
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
