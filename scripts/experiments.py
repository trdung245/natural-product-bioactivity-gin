"""Extra experiments and ablations beyond the core A/B comparison.

Subcommands:
  multiseed            Re-run A / B / B+FP across seeds; report mean +/- std.
  extract-chembl-full  Extract the FULL antiparasitic ChEMBL set (not COCONUT-restricted).
  build-decoupled      Build a broad-ChEMBL training pool, scaffold-clean vs the NP test set.
  finetune-decoupled   Train on broad ChEMBL, select+test on natural products only.
  ensemble             Combine a GIN with the RF baseline (average / logistic stack).
  explain              Per-atom activity attribution (2D + 3D renders) for demo molecules.

Examples:
    python scripts/experiments.py multiseed --seeds 42 1 7
    python scripts/experiments.py extract-chembl-full --chembl-db /path/to/chembl_37.db
    python scripts/experiments.py build-decoupled --cap 100000
    python scripts/experiments.py finetune-decoupled --epochs 15
    python scripts/experiments.py ensemble --device cpu --gin strategy_b_gmae
    python scripts/experiments.py explain --model strategy_b_gmae
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.data import (
    build_graph_list,
    get_scaffold,
    load_graphs,
    morgan_fp,
    save_graphs,
)
from src.metrics import collect_predictions
from src.train import (
    add_common_args,
    build_classifier,
    finetune,
    get_device,
    load_config,
    set_seed,
)

csv.field_size_limit(sys.maxsize)

MACRO = ["macro_AUROC", "macro_AUPR", "macro_F1"]
# (tag, strategy, use_fp) — the three headline models
HEADLINE = [
    ("strategy_a", "A", False),
    ("strategy_b", "B", False),
    ("strategy_b_fp", "B", True),
]
PATHOGENS = {
    "antimalarial": "%Plasmodium%",
    "antitrypanosomal": "%Trypanosoma%",
    "antileishmanial": "%Leishmania%",
    "antitubercular": "%Mycobacterium tuberculosis%",
}
LABELS = list(PATHOGENS)
ACTIVE_PCHEMBL = 5.0
GRID = np.arange(0.05, 0.96, 0.01)


# =============================================================================
# multiseed — mean +/- std of A / B / B+FP over several seeds
# =============================================================================
def cmd_multiseed(args):
    cfg = load_config(args.config)
    device = get_device(args.device)
    label_names = list(cfg.labels)

    records = []  # one row per (model, seed)
    for tag, strategy, use_fp in HEADLINE:
        if strategy == "A":
            pretrained, lr = None, cfg.finetune.lr_strategy_a
        else:
            pretrained = str(Path(cfg.paths.checkpoint_dir) / "gin_pretrained.pt")
            lr = cfg.finetune.lr_strategy_b

        for seed in args.seeds:
            set_seed(seed)
            print(f"\n===== {tag} | seed {seed} =====")
            _, _, test_metrics = finetune(
                cfg, device, pretrained, lr, label_names,
                use_masked_loss=True, use_fp=use_fp,
            )
            records.append({"model": tag, "seed": seed, **test_metrics})
            print(f"  {tag} seed{seed}: "
                  + "  ".join(f"{m}={test_metrics[m]:.3f}" for m in MACRO))

    df = pd.DataFrame(records)
    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "multiseed_runs.csv", index=False)

    summary = {}
    for tag in df["model"].unique():
        sub = df[df["model"] == tag]
        summary[tag] = {}
        for m in MACRO:
            summary[tag][f"{m}_mean"] = float(sub[m].mean())
            summary[tag][f"{m}_std"] = float(sub[m].std(ddof=1)) if len(sub) > 1 else 0.0

    pd.DataFrame(summary).T.to_csv(res_dir / "multiseed_summary.csv")
    with open(res_dir / "multiseed_summary.json", "w") as f:
        json.dump({"seeds": args.seeds, "summary": summary}, f, indent=2)

    print("\n=== Multi-seed summary (mean +/- std over seeds) ===")
    for tag, s in summary.items():
        print(f"{tag:<16} "
              + "  ".join(f"{m}={s[m+'_mean']:.3f}±{s[m+'_std']:.3f}" for m in MACRO))
    print(f"\nWrote {res_dir/'multiseed_summary.csv'}")


# =============================================================================
# extract-chembl-full — every ChEMBL antiparasitic compound + its SMILES
# =============================================================================
def cmd_extract_chembl_full(args):
    conn = sqlite3.connect(args.chembl_db)
    case = "\n".join(f"WHEN assay_organism LIKE '{p}' THEN '{l}'"
                     for l, p in PATHOGENS.items())
    where = " OR ".join(f"assay_organism LIKE '{p}'" for p in PATHOGENS.values())
    conn.executescript(f"""
        DROP TABLE IF EXISTS assay_label;
        CREATE TEMP TABLE assay_label AS
            SELECT assay_id, CASE {case} END AS lab FROM assays WHERE {where};
        CREATE INDEX tmp_al_assay ON assay_label(assay_id);
    """)

    sql = f"""
        SELECT cs.standard_inchi_key AS k, MIN(cs.canonical_smiles) AS smi,
               al.lab AS lab,
               MAX(CASE WHEN (act.pchembl_value IS NOT NULL AND act.pchembl_value >= {ACTIVE_PCHEMBL})
                             OR act.activity_comment IN ('active','Active')
                        THEN 1 ELSE 0 END) AS active
        FROM assay_label al
        JOIN activities act ON act.assay_id = al.assay_id
        JOIN compound_structures cs ON cs.molregno = act.molregno
        WHERE cs.standard_inchi_key IS NOT NULL AND cs.canonical_smiles IS NOT NULL
        GROUP BY cs.standard_inchi_key, al.lab
    """
    # inchikey -> {smiles, label -> 0/1}
    rows: dict[str, dict] = {}
    for key, smi, lab, active in conn.execute(sql):
        r = rows.setdefault(key, {"smiles": smi})
        r[lab] = 1 if active else 0
    conn.close()

    counts = {lab: [0, 0] for lab in LABELS}  # [active, inactive]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inchikey", "canonical_smiles", *LABELS])
        for key, r in rows.items():
            cells = [key, r["smiles"]]
            for lab in LABELS:
                v = r.get(lab, "")
                cells.append(v)
                if v == 1:
                    counts[lab][0] += 1
                elif v == 0:
                    counts[lab][1] += 1
            w.writerow(cells)

    for lab in LABELS:
        a, i = counts[lab]
        print(f"{lab:<18} active={a:>7,}  inactive={i:>7,}  tested={a+i:>8,}")
    print(f"Wrote {len(rows):,} labeled ChEMBL compounds -> {args.out}")


# =============================================================================
# build-decoupled — broad-ChEMBL training pool, scaffold-cleaned vs NP test
# =============================================================================
def cmd_build_decoupled(args):
    cfg = load_config(args.config)
    seed = args.seed or cfg.seed
    set_seed(seed)
    proc = Path(cfg.paths.processed_dir)
    label_cols = list(cfg.labels)

    # 1. Scaffolds reserved by the NP held-out sets (must NOT leak into train) --
    held = []
    for split in ("labeled_val.pt", "labeled_test.pt"):
        held += [g.smiles for g in load_graphs(proc / split)]
    held_scaffolds = {get_scaffold(s) for s in held} - {None}
    print(f"NP val+test compounds: {len(held)}  -> reserved scaffolds: {len(held_scaffolds)}")

    # 2. Load the broad ChEMBL set --------------------------------------------
    df = pd.read_csv(args.full_csv)
    df = df.dropna(subset=["canonical_smiles"]).drop_duplicates("inchikey")
    print(f"ChEMBL antiparasitic compounds: {len(df):,}")

    # 3. Subsample a buffer FIRST (avoids scaffolding all 623k) ----------------
    buffer = int(args.cap * 1.3) if args.cap else len(df)
    if len(df) > buffer:
        df = df.sample(n=buffer, random_state=seed).reset_index(drop=True)

    # 4. Drop compounds whose scaffold is reserved by NP val/test --------------
    df["scaffold"] = df["canonical_smiles"].map(get_scaffold)
    before = len(df)
    df = df[~df["scaffold"].isin(held_scaffolds)].copy()
    print(f"Removed {before - len(df):,} scaffold-overlapping compounds; {len(df):,} remain")

    # 5. Trim to the cap for tractability -------------------------------------
    if args.cap and len(df) > args.cap:
        df = df.sample(n=args.cap, random_state=seed).reset_index(drop=True)
        print(f"Training pool capped at {len(df):,}")

    # 6. Featurize -> PyG graphs ----------------------------------------------
    graphs = build_graph_list(df, label_cols)
    save_graphs(graphs, proc / "decoupled_train.pt")

    # label coverage report
    Y = torch.stack([g.y.squeeze(0) for g in graphs])
    M = torch.stack([g.mask.squeeze(0) for g in graphs])
    for j, name in enumerate(label_cols):
        known = int(M[:, j].sum())
        pos = int((Y[:, j] * M[:, j]).sum())
        print(f"  {name:<18} known={known:>7,}  active={pos:>6,}")
    print("Done.")


# =============================================================================
# finetune-decoupled — train on broad ChEMBL, test on natural products
# =============================================================================
def cmd_finetune_decoupled(args):
    cfg = load_config(args.config)
    # tractability overrides for the large training pool
    cfg["finetune"]["epochs"] = args.epochs
    cfg["finetune"]["batch_size"] = args.batch
    set_seed(args.seed or cfg.seed)
    device = get_device(args.device)
    label_names = list(cfg.labels)

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    res_dir = Path(cfg.paths.results_dir)
    summary = {}

    for tag, strategy, use_fp in HEADLINE:
        if strategy == "A":
            pretrained, lr = None, cfg.finetune.lr_strategy_a
        else:
            pretrained = str(ckpt_dir / "gin_pretrained.pt")
            lr = cfg.finetune.lr_strategy_b

        print(f"\n===== {tag}_dec | strategy {strategy} | fusion {use_fp} =====")
        model, _, test_metrics = finetune(
            cfg, device, pretrained, lr, label_names,
            use_masked_loss=True, use_fp=use_fp, train_file=args.train_file,
        )
        out_tag = f"{tag}_dec"
        torch.save(model.state_dict(), ckpt_dir / f"{out_tag}.pt")
        with open(res_dir / f"{out_tag}_metrics.json", "w") as f:
            json.dump({"test": test_metrics}, f, indent=2)
        pd.DataFrame([test_metrics]).to_csv(res_dir / f"{out_tag}_test.csv", index=False)
        summary[out_tag] = {m: round(test_metrics[m], 4) for m in MACRO}
        print(f"  {out_tag}: " + "  ".join(f"{m}={test_metrics[m]:.3f}" for m in MACRO))

    with open(res_dir / "decoupled_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Decoupled (broad-ChEMBL train / NP test) ===")
    for tag, s in summary.items():
        print(f"{tag:<18} " + "  ".join(f"{m}={s[m]:.3f}" for m in MACRO))


# =============================================================================
# ensemble — GIN + Random-Forest average / logistic stack
# =============================================================================
def _split_arrays(graphs, attach_fp):
    """Fingerprint matrix X, labels y, mask m (graph order); optionally set g.fp."""
    X, Y, M = [], [], []
    for g in graphs:
        fpt = morgan_fp(g.smiles)               # (1, 2048) float tensor
        if attach_fp:
            g.fp = fpt
        X.append(fpt.squeeze(0).numpy())
        Y.append(g.y.squeeze(0).numpy())
        M.append(g.mask.squeeze(0).numpy())
    return np.stack(X), np.stack(Y), np.stack(M)


def _gin_probs(cfg, device, graphs, gin_stem, is_fusion):
    loader = DataLoader(graphs, batch_size=cfg.finetune.batch_size)   # no shuffle
    model = build_classifier(cfg, None, device, use_fp=is_fusion)
    ckpt = Path(cfg.paths.checkpoint_dir) / f"{gin_stem}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    preds, _, _ = collect_predictions(model, loader, device)
    return preds                                                      # (n, L)


def _train_rf(Xtr, ytr, mtr, n_labels, seed):
    models = {}
    for j in range(n_labels):
        tr = mtr[:, j].astype(bool)
        if tr.sum() == 0 or len(np.unique(ytr[tr, j])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
        clf.fit(Xtr[tr], ytr[tr, j])
        models[j] = clf
    return models


def _rf_probs(models, X, n_labels):
    P = np.full((X.shape[0], n_labels), np.nan)
    for j, clf in models.items():
        P[:, j] = clf.predict_proba(X)[:, 1]
    return P


def _best_thresholds(pv, yv, mv, n):
    thr = np.full(n, 0.5)
    for j in range(n):
        mm = mv[:, j].astype(bool)
        if mm.sum() == 0 or len(np.unique(yv[mm, j])) < 2:
            continue
        yt, yp = yv[mm, j], pv[mm, j]
        thr[j] = float(GRID[int(np.argmax(
            [f1_score(yt, (yp >= t).astype(int), zero_division=0) for t in GRID]))])
    return thr


def _evaluate(pt, yt, mt, pv, yv, mv, n):
    """Macro AUROC / AUPR (test) and macro-F1 at per-label val-tuned thresholds."""
    thr = _best_thresholds(pv, yv, mv, n)
    aurocs, auprs, f1s = [], [], []
    for j in range(n):
        mm = mt[:, j].astype(bool)
        if mm.sum() == 0:
            continue
        yy, pp = yt[mm, j], pt[mm, j]
        if len(np.unique(yy)) >= 2:
            aurocs.append(roc_auc_score(yy, pp))
        auprs.append(average_precision_score(yy, pp))
        f1s.append(f1_score(yy, (pp >= thr[j]).astype(int), zero_division=0))
    return (float(np.mean(aurocs)), float(np.mean(auprs)), float(np.mean(f1s)))


def _stack(pv_g, pv_r, yv, mv, pt_g, pt_r, n):
    """Per-label logistic stack: fit on val [gin, rf], predict test."""
    out = np.full_like(pt_g, np.nan)
    for j in range(n):
        mm = mv[:, j].astype(bool)
        if mm.sum() == 0 or len(np.unique(yv[mm, j])) < 2:
            out[:, j] = 0.5 * (pt_g[:, j] + pt_r[:, j])   # fallback
            continue
        Xv = np.column_stack([pv_g[mm, j], pv_r[mm, j]])
        lr = LogisticRegression(max_iter=1000).fit(Xv, yv[mm, j])
        Xt = np.column_stack([pt_g[:, j], pt_r[:, j]])
        out[:, j] = lr.predict_proba(Xt)[:, 1]
    return out


def cmd_ensemble(args):
    cfg = load_config(args.config)
    seed = args.seed or cfg.seed
    set_seed(seed)
    device = get_device(args.device)
    labels = list(cfg.labels)
    n = len(labels)
    is_fusion = args.fusion or args.gin.endswith("_fp") or args.gin.endswith("_fp_gmae")
    proc = Path(cfg.paths.processed_dir)

    train = load_graphs(proc / "labeled_train.pt")
    val = load_graphs(proc / "labeled_val.pt")
    test = load_graphs(proc / "labeled_test.pt")

    Xtr, ytr, mtr = _split_arrays(train, attach_fp=False)
    Xv, yv, mv = _split_arrays(val, attach_fp=is_fusion)
    Xt, yt, mt = _split_arrays(test, attach_fp=is_fusion)

    print(f"Training RF (per label) ... GIN partner = {args.gin} (fusion={is_fusion})")
    rf = _train_rf(Xtr, ytr, mtr, n, seed)
    rf_v, rf_t = _rf_probs(rf, Xv, n), _rf_probs(rf, Xt, n)
    gin_v = _gin_probs(cfg, device, val, args.gin, is_fusion)
    gin_t = _gin_probs(cfg, device, test, args.gin, is_fusion)

    avg_v, avg_t = 0.5 * (gin_v + rf_v), 0.5 * (gin_t + rf_t)
    stk_t = _stack(gin_v, rf_v, yv, mv, gin_t, rf_t, n)
    stk_v = _stack(gin_v, rf_v, yv, mv, gin_v, rf_v, n)  # val stack (for its own thresholds)

    results = {
        f"GIN ({args.gin})": _evaluate(gin_t, yt, mt, gin_v, yv, mv, n),
        "RF":                _evaluate(rf_t, yt, mt, rf_v, yv, mv, n),
        "ensemble AVG":      _evaluate(avg_t, yt, mt, avg_v, yv, mv, n),
        "ensemble STACK":    _evaluate(stk_t, yt, mt, stk_v, yv, mv, n),
    }

    print(f"\n{'model':22s} {'AUROC':>7} {'AUPR':>7} {'F1*':>7}   (*=per-label tuned)")
    for name, (a, p, f) in results.items():
        print(f"{name:22s} {a:7.3f} {p:7.3f} {f:7.3f}")

    out = {name: {"macro_AUROC": a, "macro_AUPR": p, "macro_F1_tuned": f}
           for name, (a, p, f) in results.items()}
    res_path = Path(cfg.paths.results_dir) / f"ensemble_{args.gin}.json"
    with open(res_path, "w") as fh:
        json.dump({"gin": args.gin, "results": out}, fh, indent=2)
    print(f"\nWrote {res_path}")


# =============================================================================
# explain — per-atom activity attribution (2D + 3D renders)
# =============================================================================
# For a GIN with sum pooling the graph embedding is h_G = Σ_v z_v, so each atom's
# contribution to a label's pre-sigmoid logit is exactly  s_v = g · z_v  with
# g = ∂logit/∂h_G (a first-order / CAM-style decomposition: Σ_v s_v = g · h_G).
# Positive s_v ⇒ the atom pushes the prediction toward "active".
DEMO = {
    "quinine": "C=CC1CN2CCC1CC2C(O)c1ccnc2ccc(OC)cc12",          # antimalarial
    "artemisinin": "CC1CCC2C(C)C(=O)OC3OC4(C)CCC1C23OO4",        # antimalarial
    "chloroquine": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",          # antimalarial
}


def _atom_contributions(model, smiles, label_idx):
    from src.data import smiles_to_graph
    data = smiles_to_graph(smiles)
    z = model.encoder.forward_nodes(data.x, data.edge_index, data.edge_attr)
    h_g = z.sum(0, keepdim=True)
    logit = model.head(h_g)                       # (1, num_labels), pre-sigmoid
    probs = torch.sigmoid(logit)[0].detach().numpy()
    g = torch.autograd.grad(logit[0, label_idx], z)[0]
    s = (z * g).sum(1).detach().numpy()
    return probs, s


def cmd_explain(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    cmap = cm.get_cmap("PiYG")   # green = drives activity, pink = pushes against

    def atom_colors(weights):
        lim = float(np.abs(weights).max()) or 1.0
        norm = mcolors.Normalize(vmin=-lim, vmax=lim)
        return {i: cmap(norm(w))[:3] for i, w in enumerate(weights)}, lim

    def render_2d(smiles, weights, title, out):
        mol = Chem.MolFromSmiles(smiles)
        colors, _ = atom_colors(weights)
        d = rdMolDraw2D.MolDraw2DCairo(560, 460)
        d.drawOptions().addAtomIndices = False
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, highlightAtoms=list(range(mol.GetNumAtoms())),
            highlightAtomColors=colors, highlightBonds=[])
        d.FinishDrawing()
        Path(out).write_bytes(d.GetDrawingText())

    def render_3d_png(smiles, weights, title, out):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        n_heavy = len(weights)
        pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
        colors, lim = atom_colors(weights)

        fig = plt.figure(figsize=(6.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ax.plot(*zip(pos[i], pos[j]), color="#888888", lw=1.2, zorder=1)
        for i in range(mol.GetNumAtoms()):
            if i < n_heavy:
                c = colors[i]; size = 260; edge = "black"
            else:
                c = (0.8, 0.8, 0.8); size = 60; edge = "none"   # hydrogens
            ax.scatter(*pos[i], color=c, s=size, edgecolors=edge, linewidths=0.6,
                       depthshade=True, zorder=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_axis_off()
        sm = cm.ScalarMappable(norm=mcolors.Normalize(-lim, lim), cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("atom contribution  (green → active)", fontsize=9)
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)

    def render_3d_html(smiles, weights, title, out):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
        molblock = Chem.MolToMolBlock(mol)
        colors, lim = atom_colors(weights)
        hexcol = {i: mcolors.to_hex(c) for i, c in colors.items()}
        style_js = "\n".join(
            f"viewer.setStyle({{index:{i}}}, {{stick:{{color:'{h}'}},"
            f"sphere:{{color:'{h}',scale:0.30}}}});" for i, h in hexcol.items())
        html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>body{{font-family:sans-serif;margin:0;background:#fafafa}}
#h{{padding:10px 16px}} #v{{width:100vw;height:80vh;position:relative}}
.bar{{height:16px;width:320px;background:linear-gradient(90deg,#c51b7d,#f7f7f7,#4d9221);
border:1px solid #ccc;border-radius:3px}}</style></head><body>
<div id="h"><b>{title}</b><br><small>drag to rotate · scroll to zoom · atoms coloured by
contribution to the predicted activity</small>
<div style="display:flex;gap:8px;align-items:center;margin-top:6px">
<small>against</small><div class="bar"></div><small>drives activity</small></div></div>
<div id="v"></div>
<script>
let viewer = $3Dmol.createViewer("v", {{backgroundColor:"white"}});
viewer.addModel(`{molblock}`, "mol");
{style_js}
viewer.zoomTo(); viewer.render();
</script></body></html>"""
        Path(out).write_text(html)

    cfg = load_config(args.config)
    device = torch.device("cpu")
    labels = list(cfg.labels)

    model = build_classifier(cfg, None, device, use_fp=False)
    model.load_state_dict(torch.load(
        Path(cfg.paths.checkpoint_dir) / f"{args.model}.pt", map_location=device))
    model.eval()

    figdir = Path(cfg.paths.results_dir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    j = labels.index(args.label)
    for name, smiles in DEMO.items():
        probs, s = _atom_contributions(model, smiles, j)
        pred = ", ".join(f"{l} {probs[i]:.2f}" for i, l in enumerate(labels))
        title = f"{name} — {args.label} (p={probs[j]:.2f})"
        print(f"\n{name}: {pred}\n  -> explaining '{args.label}'; "
              f"top atoms {np.argsort(-s)[:4].tolist()} (most 'active'-driving)")
        render_2d(smiles, s, title, figdir / f"explain_{name}_2d.png")
        render_3d_png(smiles, s, title, figdir / f"explain_{name}_3d.png")
        render_3d_html(smiles, s, title, figdir / f"explain_{name}_3d.html")

    print(f"\nWrote explain_*_2d.png / _3d.png / _3d.html to {figdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = add_common_args(sub.add_parser("multiseed", help="mean +/- std over seeds"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    p.set_defaults(func=cmd_multiseed)

    p = sub.add_parser("extract-chembl-full", help="full ChEMBL antiparasitic set")
    p.add_argument("--chembl-db", required=True)
    p.add_argument("--out", default="data/raw/chembl_full_labels.csv")
    p.set_defaults(func=cmd_extract_chembl_full)

    p = add_common_args(sub.add_parser("build-decoupled", help="broad-ChEMBL train pool"))
    p.add_argument("--full-csv", default="data/raw/chembl_full_labels.csv")
    p.add_argument("--cap", type=int, default=100_000,
                   help="Max training compounds after scaffold filtering.")
    p.set_defaults(func=cmd_build_decoupled)

    p = add_common_args(sub.add_parser("finetune-decoupled", help="train broad, test NP"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--train-file", default="decoupled_train.pt")
    p.set_defaults(func=cmd_finetune_decoupled)

    p = add_common_args(sub.add_parser("ensemble", help="GIN + RF average / stack"))
    p.add_argument("--gin", default="strategy_b_gmae",
                   help="GIN checkpoint stem to ensemble with RF.")
    p.add_argument("--fusion", action="store_true",
                   help="Set if the --gin checkpoint is a fusion (_fp) model.")
    p.set_defaults(func=cmd_ensemble)

    p = add_common_args(sub.add_parser("explain", help="per-atom activity attribution"))
    p.add_argument("--model", default="strategy_b_gmae",
                   help="Pure-GIN checkpoint stem (no fusion, for clean attribution).")
    p.add_argument("--label", default="antimalarial",
                   help="Which activity to explain (demo molecules are antimalarials).")
    p.set_defaults(func=cmd_explain)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
