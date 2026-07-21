"""Evaluation metrics for multi-label bioactivity prediction (mask-aware)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader and return (preds, labels, masks) arrays."""
    model.eval()
    preds, labels, masks = [], [], []
    for batch in loader:
        batch = batch.to(device)
        p = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                  getattr(batch, "fp", None))
        preds.append(p.cpu())
        labels.append(batch.y.cpu())
        masks.append(batch.mask.cpu())
    return (
        torch.cat(preds).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(masks).numpy(),
    )


def per_label_metrics(preds, labels, masks, label_names, threshold=0.5) -> dict:
    """AUROC / AUPR / F1 per label, skipping masked (unknown) entries.

    A label with only one class present among the known entries has an undefined
    AUROC; it is reported as NaN rather than crashing.
    """
    results: dict[str, dict[str, float]] = {}
    for j, name in enumerate(label_names):
        m = masks[:, j].astype(bool)
        if m.sum() == 0:
            continue
        y_true = labels[m, j]
        y_prob = preds[m, j]
        single_class = len(np.unique(y_true)) < 2
        results[name] = {
            "AUROC": float("nan") if single_class else roc_auc_score(y_true, y_prob),
            "AUPR": average_precision_score(y_true, y_prob),
            "F1": f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0),
            "n": int(m.sum()),
            "n_pos": int(y_true.sum()),
        }
    return results


def macro_f1(preds, labels, masks, label_names, threshold=0.5) -> float:
    res = per_label_metrics(preds, labels, masks, label_names, threshold)
    f1s = [v["F1"] for v in res.values()]
    return float(np.mean(f1s)) if f1s else float("nan")


def summary_row(preds, labels, masks, label_names, threshold=0.5) -> dict:
    """Flatten per-label metrics into one row plus macro averages."""
    res = per_label_metrics(preds, labels, masks, label_names, threshold)
    row: dict[str, float] = {}
    for name, m in res.items():
        row[f"{name}_AUROC"] = m["AUROC"]
        row[f"{name}_AUPR"] = m["AUPR"]
        row[f"{name}_F1"] = m["F1"]
    aurocs = [m["AUROC"] for m in res.values() if not np.isnan(m["AUROC"])]
    row["macro_AUROC"] = float(np.mean(aurocs)) if aurocs else float("nan")
    row["macro_AUPR"] = float(np.mean([m["AUPR"] for m in res.values()])) if res else float("nan")
    row["macro_F1"] = macro_f1(preds, labels, masks, label_names, threshold)
    return row
