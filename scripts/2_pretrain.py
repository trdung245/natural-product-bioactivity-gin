"""Stage 2 — Self-supervised pretraining on the unlabeled COCONUT corpus.

All three subcommands save a GINEncoder state_dict that Strategy-B fine-tuning
loads directly (pass its path to `3_finetune.py run --encoder ...`).

Subcommands:
  mask         Attribute masking — reconstruct 15%-masked atom types (Stage-1 baseline).
  graphmae     GraphMAE generative masked-autoencoder (harder pretext; streams shards).
  contrastive  MolCLR contrastive learning (two augmented views + NT-Xent).

Examples:
    python scripts/2_pretrain.py mask
    python scripts/2_pretrain.py graphmae --epochs 15          # full-corpus shards
    python scripts/2_pretrain.py graphmae --pretrain-file pretrain.pt --epochs 20
    python scripts/2_pretrain.py contrastive --epochs 12
    python scripts/2_pretrain.py contrastive --shards --epochs 6   # full 714k
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import gc
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.data import load_graphs
from src.model import (
    GINContrastive,
    GINGraphMAE,
    GINPretrain,
    augment_view,
    graphmae_mask,
    mask_atom_features,
    nt_xent_loss,
    sce_loss,
)
from src.train import add_common_args, get_device, load_config, set_seed


# =============================================================================
# mask — attribute-masking pretraining (Stage-1 baseline)
# =============================================================================
def cmd_mask(args):
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)
    device = get_device(args.device)

    graphs = load_graphs(Path(cfg.paths.processed_dir) / "pretrain.pt")
    loader = DataLoader(graphs, batch_size=cfg.pretrain.batch_size, shuffle=True)

    model = GINPretrain(
        in_dim=cfg.model.in_dim,
        edge_dim=cfg.model.edge_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_atom_types=cfg.pretrain.num_atom_types,
        dropout=cfg.model.dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.pretrain.lr)

    epochs = args.epochs or cfg.pretrain.epochs
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            masked_x, mask_index, orig_types = mask_atom_features(
                batch.x, cfg.pretrain.mask_rate, model.mask_token
            )
            optim.zero_grad()
            logits = model(masked_x, batch.edge_index, batch.edge_attr, mask_index)
            loss = torch.nn.functional.cross_entropy(logits, orig_types)
            loss.backward()
            optim.step()
            total += loss.item()
        print(f"Pretrain epoch {epoch:03d}  loss={total / len(loader):.4f}")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / args.out
    torch.save(model.encoder.state_dict(), path)
    print(f"Saved pretrained encoder -> {path}")


# =============================================================================
# graphmae — GraphMAE pretraining, streamed over shards
# =============================================================================
def _shard_paths(proc: Path, pretrain_file: str | None):
    if pretrain_file:
        return [proc / pretrain_file]
    shards = sorted(proc.glob("pretrain_shard_*.pt"))
    if not shards:
        raise FileNotFoundError(
            "No pretrain_shard_*.pt found. Run `1_prepare.py build-shards` first, "
            "or pass --pretrain-file pretrain.pt.")
    return shards


def cmd_graphmae(args):
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)
    device = get_device(args.device)

    proc = Path(cfg.paths.processed_dir)
    shards = _shard_paths(proc, args.pretrain_file)
    gm = cfg.graphmae
    epochs = args.epochs or gm.epochs
    print(f"GraphMAE | shards={len(shards)} | mask_rate={gm.mask_rate} | "
          f"gamma={gm.gamma} | dec_layers={gm.dec_layers} | epochs={epochs} | {device}")

    model = GINGraphMAE(
        in_dim=cfg.model.in_dim, edge_dim=cfg.model.edge_dim,
        hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers,
        dec_layers=gm.dec_layers, dropout=cfg.model.dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=gm.lr)

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        seen_loss, seen_batches = 0.0, 0
        for shard in shards:                       # stream: one shard in RAM at a time
            graphs = load_graphs(shard)
            loader = DataLoader(graphs, batch_size=gm.batch_size, shuffle=True)
            for batch in loader:
                batch = batch.to(device)
                masked_x, mask_index, target = graphmae_mask(
                    batch.x, gm.mask_rate, model.enc_mask_token)
                optim.zero_grad()
                recon = model(masked_x, batch.edge_index, batch.edge_attr, mask_index)
                loss = sce_loss(recon[mask_index], target, gm.gamma)
                loss.backward()
                optim.step()
                seen_loss += loss.item()
                seen_batches += 1
            del graphs, loader
            gc.collect()
        print(f"GraphMAE epoch {epoch:03d}  loss={seen_loss/max(seen_batches,1):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        # Checkpoint the encoder every epoch so an early stop still yields a
        # usable Strategy-B encoder (this run is long).
        torch.save(model.encoder.state_dict(), ckpt_dir / args.out)

    print(f"Saved GraphMAE encoder -> {ckpt_dir / args.out}")


# =============================================================================
# contrastive — MolCLR-style contrastive pretraining
# =============================================================================
def cmd_contrastive(args):
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)
    device = get_device(args.device)

    proc = Path(cfg.paths.processed_dir)
    shards = (sorted(proc.glob("pretrain_shard_*.pt")) if args.shards
              else [proc / args.pretrain_file])
    cc = cfg.contrastive
    epochs = args.epochs or cc.epochs
    print(f"MolCLR | sources={len(shards)} | node_mask={cc.node_mask_rate} | "
          f"edge_drop={cc.edge_drop_rate} | temp={cc.temperature} | "
          f"proj={cc.proj_dim} | epochs={epochs} | {device}")

    model = GINContrastive(
        in_dim=cfg.model.in_dim, edge_dim=cfg.model.edge_dim,
        hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers,
        proj_dim=cc.proj_dim, dropout=cfg.model.dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cc.lr)

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / (Path(args.out).stem + ".resume.pt")

    start_epoch = 0
    if args.resume and resume_path.exists():
        st = torch.load(resume_path, map_location=device)
        model.load_state_dict(st["model"])
        optim.load_state_dict(st["optim"])
        start_epoch = st["epoch"] + 1
        print(f"Resumed from {resume_path}: continuing at epoch {start_epoch}/{epochs}",
              flush=True)
    elif args.warm_start:
        model.encoder.load_state_dict(torch.load(args.warm_start, map_location=device))
        print(f"Warm-started encoder from {args.warm_start} (projection + optim fresh)",
              flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for src in shards:
            graphs = load_graphs(src)
            loader = DataLoader(graphs, batch_size=cc.batch_size, shuffle=True)
            for batch in loader:
                batch = batch.to(device)
                x1, ei1, ea1 = augment_view(batch.x, batch.edge_index, batch.edge_attr,
                                            model.mask_token, cc.node_mask_rate, cc.edge_drop_rate)
                x2, ei2, ea2 = augment_view(batch.x, batch.edge_index, batch.edge_attr,
                                            model.mask_token, cc.node_mask_rate, cc.edge_drop_rate)
                optim.zero_grad()
                z1 = model(x1, ei1, ea1, batch.batch)
                z2 = model(x2, ei2, ea2, batch.batch)
                loss = nt_xent_loss(z1, z2, cc.temperature)
                loss.backward()
                optim.step()
                tot += loss.item()
                nb += 1
            del graphs, loader
            gc.collect()
        print(f"MolCLR epoch {epoch:03d}  loss={tot/max(nb,1):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.encoder.state_dict(), ckpt_dir / args.out)
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optim": optim.state_dict()}, resume_path)

    print(f"Saved MolCLR encoder -> {ckpt_dir / args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = add_common_args(sub.add_parser("mask", help="attribute-masking pretraining"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", default="gin_pretrained.pt")
    p.set_defaults(func=cmd_mask)

    p = add_common_args(sub.add_parser("graphmae", help="GraphMAE pretraining"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pretrain-file", default=None,
                   help="Single .pt in processed_dir instead of shards.")
    p.add_argument("--out", default="gin_graphmae.pt")
    p.set_defaults(func=cmd_graphmae)

    p = add_common_args(sub.add_parser("contrastive", help="MolCLR contrastive pretraining"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pretrain-file", default="pretrain.pt",
                   help="Single .pt in processed_dir (default 200k subsample).")
    p.add_argument("--shards", action="store_true",
                   help="Stream pretrain_shard_*.pt (full 714k) instead.")
    p.add_argument("--out", default="gin_molclr.pt")
    p.add_argument("--warm-start", default=None,
                   help="Init encoder weights from this .pt (fresh projection + optim).")
    p.add_argument("--resume", action="store_true",
                   help="Resume full model+optimizer from <out>.resume.pt if present.")
    p.set_defaults(func=cmd_contrastive)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
