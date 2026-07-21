"""Config, run utilities, and the shared supervised fine-tuning loop.

Groups three concerns that every script needs:
  * config   — a tiny YAML loader with dotted access + shared CLI args;
  * runtime  — reproducible seeding and device selection;
  * finetune — the Strategy-A / Strategy-B training loop (masked or naive loss,
               with an optional Morgan-fingerprint fusion branch).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from .data import FP_BITS, compute_pos_weight, load_graphs, morgan_fp
from .metrics import collect_predictions, summary_row
from .model import GINClassifier, masked_bce_loss, naive_bce_loss


# =============================================================================
# Config
# =============================================================================
class Config(dict):
    """Dict that also supports attribute access and nested dicts."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    with open(path, "r") as f:
        return Config(yaml.safe_load(f))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Args shared by every script / subcommand."""
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the random seed.")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu | mps. Auto-detected if omitted.")
    return parser


# =============================================================================
# Runtime utilities
# =============================================================================
def set_seed(seed: int = 42) -> None:
    """Make a run reproducible across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str | None = None) -> torch.device:
    """Pick a device, honoring an explicit request when given."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Supervised fine-tuning loop
# =============================================================================
def _attach_fingerprints(graphs):
    """Add a Morgan fingerprint (data.fp, shape (1, FP_BITS)) to each graph.

    PyG concatenates it along dim 0 when batching, like ``y`` — so the loop sees
    ``batch.fp`` of shape (num_graphs, FP_BITS).
    """
    for g in graphs:
        g.fp = morgan_fp(g.smiles)


def build_loaders(processed_dir, batch_size, use_fp=False, train_file="labeled_train.pt"):
    proc = Path(processed_dir)
    train = load_graphs(proc / train_file)
    val = load_graphs(proc / "labeled_val.pt")
    test = load_graphs(proc / "labeled_test.pt")
    if use_fp:
        for split in (train, val, test):
            _attach_fingerprints(split)
    return (
        train,
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


def build_classifier(cfg, pretrained_encoder: str | None, device, use_fp=False):
    model = GINClassifier(
        in_dim=cfg.model.in_dim,
        edge_dim=cfg.model.edge_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
        fp_dim=FP_BITS if use_fp else 0,
    ).to(device)
    if pretrained_encoder:
        state = torch.load(pretrained_encoder, map_location=device)
        model.encoder.load_state_dict(state)
        print(f"Loaded pretrained encoder from {pretrained_encoder}")
    return model


def finetune(
    cfg,
    device,
    pretrained_encoder: str | None,
    lr: float,
    label_names: list[str],
    use_masked_loss: bool = True,
    use_fp: bool = False,
    train_file: str = "labeled_train.pt",
):
    """Train a GINClassifier and return (model, best_val_metrics, test_metrics)."""
    train_graphs, train_loader, val_loader, test_loader = build_loaders(
        cfg.paths.processed_dir, cfg.finetune.batch_size, use_fp=use_fp,
        train_file=train_file,
    )

    model = build_classifier(cfg, pretrained_encoder, device, use_fp=use_fp)
    optim = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=cfg.finetune.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=cfg.finetune.patience
    )

    pos_weight = None
    if cfg.finetune.use_pos_weight:
        pos_weight = compute_pos_weight(train_graphs, cfg.model.num_labels).to(device)

    loss_fn = masked_bce_loss if use_masked_loss else naive_bce_loss

    best_val_macro = -1.0
    best_state = None
    best_val_metrics = None

    for epoch in range(cfg.finetune.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                         getattr(batch, "fp", None))
            loss = loss_fn(pred, batch.y, batch.mask, pos_weight)
            loss.backward()
            optim.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        scheduler.step(train_loss)

        preds, labels, masks = collect_predictions(model, val_loader, device)
        val = summary_row(preds, labels, masks, label_names, cfg.eval.threshold)
        if val["macro_AUROC"] > best_val_macro:
            best_val_macro = val["macro_AUROC"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val
        print(f"Epoch {epoch:03d}  loss={train_loss:.4f}  "
              f"val_macroAUROC={val['macro_AUROC']:.3f}  val_macroF1={val['macro_F1']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    preds, labels, masks = collect_predictions(model, test_loader, device)
    test_metrics = summary_row(preds, labels, masks, label_names, cfg.eval.threshold)
    return model, best_val_metrics, test_metrics
