"""Stage 3 — Supervised fine-tuning (Strategy A / B).

  Strategy A : random-init GIN, ChEMBL labels only (baseline).
  Strategy B : load the Stage-2 pretrained encoder, then fine-tune (proposed).
  --naive-loss : ablation — treat missing labels as inactive instead of masking.
  --fusion     : fuse a Morgan fingerprint with the graph embedding (GNN+FP).
  --encoder    : Strategy B only — use a specific pretrained encoder (e.g. GraphMAE).

Examples:
    python scripts/3_finetune.py --strategy A
    python scripts/3_finetune.py --strategy B
    python scripts/3_finetune.py --strategy B --naive-loss
    python scripts/3_finetune.py --strategy B --encoder checkpoints/gin_graphmae.pt --tag strategy_b_gmae
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.train import add_common_args, finetune, get_device, load_config, set_seed


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--strategy", choices=["A", "B"], required=True)
    parser.add_argument("--naive-loss", action="store_true",
                        help="Treat missing labels as inactive (ablation).")
    parser.add_argument("--fusion", action="store_true",
                        help="Fuse a Morgan fingerprint with the graph embedding.")
    parser.add_argument("--encoder", default=None,
                        help="Strategy B only: path to a pretrained encoder to load "
                             "instead of checkpoints/gin_pretrained.pt (e.g. GraphMAE).")
    parser.add_argument("--tag", default=None, help="Override the output name.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed or cfg.seed)
    device = get_device(args.device)
    label_names = list(cfg.labels)

    if args.strategy == "A":
        pretrained, lr = None, cfg.finetune.lr_strategy_a
    else:
        pretrained = args.encoder or str(Path(cfg.paths.checkpoint_dir) / "gin_pretrained.pt")
        lr = cfg.finetune.lr_strategy_b

    use_masked = not args.naive_loss
    print(f"Strategy {args.strategy} | lr={lr} | "
          f"loss={'masked' if use_masked else 'naive'} | "
          f"fusion={args.fusion} | device={device}")

    model, val_metrics, test_metrics = finetune(
        cfg, device, pretrained, lr, label_names, use_masked_loss=use_masked,
        use_fp=args.fusion,
    )

    tag = args.tag or (f"strategy_{args.strategy.lower()}"
                       + ("" if use_masked else "_naive")
                       + ("_fp" if args.fusion else ""))

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{tag}.pt")

    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / f"{tag}_metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    pd.DataFrame([test_metrics]).to_csv(res_dir / f"{tag}_test.csv", index=False)

    print("\n=== TEST metrics ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved checkpoint -> {ckpt_dir / (tag + '.pt')}")


if __name__ == "__main__":
    main()
