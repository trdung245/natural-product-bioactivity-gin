"""GIN encoder, task heads, and all training losses (proposal Section 3.2 / 3.6).

The backbone is a 5-layer Graph Isomorphism Network with hidden dim 300. Bond
features are folded into message passing via a per-layer edge projection (PyG
``GINEConv``); graph readout is sum pooling. The *same* encoder backs every stage,
so pretrained weights transfer to the classifier unchanged:

  Encoder      : GINEncoder            — shared backbone
  Stage 1 (SSL): GINPretrain           — attribute masking (reconstruct atom type)
                 GINGraphMAE           — GraphMAE generative masked autoencoder
                 GINContrastive        — MolCLR contrastive (two augmented views)
  Stage 2      : GINClassifier         — multi-label sigmoid head (+ optional FP)

Losses for each objective live at the bottom of the file.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

from .data import NUM_ATOM_TYPES

# =============================================================================
# Encoder
# =============================================================================
class GINEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 42,
        edge_dim: int = 10,
        hidden_dim: int = 300,
        num_layers: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.edge_encoders = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # train_eps=True learns the epsilon in (1 + eps) * h_v.
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.edge_encoders.append(nn.Linear(edge_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward_nodes(self, x, edge_index, edge_attr) -> torch.Tensor:
        """Per-atom embeddings after all message-passing layers."""
        h = self.input_proj(x)
        for conv, edge_enc, bn in zip(self.convs, self.edge_encoders, self.bns):
            e = edge_enc(edge_attr)
            h = conv(h, edge_index, e)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        """Graph-level embedding via sum pooling."""
        h = self.forward_nodes(x, edge_index, edge_attr)
        return global_add_pool(h, batch)


# =============================================================================
# Stage 2 — supervised classifier
# =============================================================================
class GINClassifier(nn.Module):
    """Stage 2: multi-label bioactivity prediction."""

    def __init__(
        self,
        in_dim: int = 42,
        edge_dim: int = 10,
        hidden_dim: int = 300,
        num_layers: int = 5,
        num_labels: int = 5,
        dropout: float = 0.5,
        fp_dim: int = 0,
    ):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim, num_layers, dropout)
        self.fp_dim = fp_dim
        head_in = hidden_dim
        if fp_dim:
            # Fuse a fixed Morgan fingerprint with the learned graph embedding
            # (FH-GNN / hybrid fingerprint-graph trend, 2024-25).
            self.fp_proj = nn.Sequential(
                nn.Linear(fp_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            head_in += hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

    def forward(self, x, edge_index, edge_attr, batch, fp=None) -> torch.Tensor:
        h = self.encoder(x, edge_index, edge_attr, batch)
        if self.fp_dim and fp is not None:
            h = torch.cat([h, self.fp_proj(fp)], dim=1)
        return torch.sigmoid(self.head(h))   # (num_graphs, num_labels)


# =============================================================================
# Stage 1 — self-supervised pretext heads
# =============================================================================
class GINPretrain(nn.Module):
    """Stage 1: attribute-masking pretraining.

    Masked atom rows are replaced by a learnable [MASK] embedding before
    encoding; a linear head then reconstructs the original atom type from each
    masked atom's contextual embedding.
    """

    def __init__(
        self,
        in_dim: int = 42,
        edge_dim: int = 10,
        hidden_dim: int = 300,
        num_layers: int = 5,
        num_atom_types: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim, num_layers, dropout)
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.pred_head = nn.Linear(hidden_dim, num_atom_types)

    def forward(self, x, edge_index, edge_attr, mask_index) -> torch.Tensor:
        """Logits over atom types for the masked atoms.

        ``x`` already has the masked rows overwritten with ``mask_token`` by the
        caller; ``mask_index`` indexes those rows in the batched node tensor.
        """
        h = self.encoder.forward_nodes(x, edge_index, edge_attr)
        return self.pred_head(h[mask_index])


class GINGraphMAE(nn.Module):
    """Stage 1 (stronger pretext) — GraphMAE (Hou et al. 2022).

    A generative masked-autoencoder for graphs. Improvements over plain attribute
    masking (``GINPretrain``):
      * higher mask rate (default 0.5) — a much harder pretext;
      * **re-mask decoding**: masked nodes' *latent* codes are overwritten by a
        learnable decoder token before decoding, so the decoder can't just copy
        the encoder's leaked signal;
      * a **GNN decoder** (not a linear head) reconstructs the *full* 42-dim atom
        feature vector, not just the 10-way atom type;
      * a **scaled cosine error** objective (see ``sce_loss``).

    The ``encoder`` is the same :class:`GINEncoder` used downstream, so its weights
    transfer to Stage 2 exactly like the attribute-masking encoder. The decoder is
    discarded after pretraining.
    """

    def __init__(
        self,
        in_dim: int = 42,
        edge_dim: int = 10,
        hidden_dim: int = 300,
        num_layers: int = 5,
        dec_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim, num_layers, dropout)
        self.enc_mask_token = nn.Parameter(torch.zeros(in_dim))
        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dec_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        # Shallow GNN decoder: latent (hidden) -> hidden -> reconstructed features.
        self.decoder = GINEncoder(hidden_dim, edge_dim, hidden_dim, dec_layers, dropout)
        self.recon_head = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, edge_index, edge_attr, mask_index) -> torch.Tensor:
        """Reconstruct atom features for every node (loss taken on masked ones).

        ``x`` already has the masked rows overwritten with ``enc_mask_token``.
        """
        h = self.encoder.forward_nodes(x, edge_index, edge_attr)   # (N, hidden)
        z = self.encoder_to_decoder(h)
        # Re-mask: replace masked nodes' latent codes with the decoder token.
        z = z.clone()
        z[mask_index] = self.dec_mask_token
        g = self.decoder.forward_nodes(z, edge_index, edge_attr)   # (N, hidden)
        return self.recon_head(g)                                  # (N, in_dim)


class GINContrastive(nn.Module):
    """Stage 1 (contrastive pretext) — MolCLR (Wang et al. 2022).

    Two stochastically augmented views of each molecule (atom masking + bond
    deletion) are encoded and pushed together, while different molecules in the
    batch are pushed apart, via the NT-Xent loss (see ``nt_xent_loss``). A small
    projection head maps the graph embedding to the space where the contrast is
    computed and is discarded afterwards; the ``encoder`` transfers to Stage 2.
    """

    def __init__(
        self,
        in_dim: int = 42,
        edge_dim: int = 10,
        hidden_dim: int = 300,
        num_layers: int = 5,
        proj_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim, num_layers, dropout)
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        """Projected graph embedding for one (already-augmented) view."""
        h = self.encoder(x, edge_index, edge_attr, batch)   # (num_graphs, hidden)
        return self.proj(h)                                 # (num_graphs, proj_dim)


# =============================================================================
# Losses & masking utilities
# =============================================================================
def masked_bce_loss(y_pred, y_true, mask, pos_weight=None) -> torch.Tensor:
    """Masked binary cross-entropy (proposal Section 3.6).

    Only labels with known assay data (``mask == True``) contribute. Positive
    examples are optionally up-weighted by per-label inverse class frequency.

    Args:
        y_pred:     (B, L) predicted probabilities.
        y_true:     (B, L) ground-truth labels in {0, 1}.
        mask:       (B, L) True where the label is known.
        pos_weight: (L,)   weight applied to positive entries.
    """
    loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    if pos_weight is not None:
        w = torch.where(y_true.bool(), pos_weight.unsqueeze(0), torch.ones_like(y_true))
        loss = loss * w
    mask = mask.float()
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def naive_bce_loss(y_pred, y_true, mask, pos_weight=None) -> torch.Tensor:
    """Ablation loss: treat unknown labels as inactive (no masking).

    Isolates the value of correct missing-label handling (proposal Section 5,
    outcome 3 / Phase 8). The mask argument is accepted but ignored so this is
    a drop-in replacement for ``masked_bce_loss``.
    """
    y_true_naive = y_true * mask.float()   # unknown -> 0 (inactive)
    loss = F.binary_cross_entropy(y_pred, y_true_naive, reduction="none")
    if pos_weight is not None:
        w = torch.where(y_true_naive.bool(), pos_weight.unsqueeze(0), torch.ones_like(y_true_naive))
        loss = loss * w
    return loss.mean()


def sce_loss(recon, target, gamma: float = 2.0) -> torch.Tensor:
    """Scaled cosine error (GraphMAE, Hou et al. 2022).

        L = mean( (1 − cos(recon, target))^gamma )

    Cosine (not MSE) makes the objective scale-invariant and robust for the
    mostly-binary one-hot atom features; the exponent ``gamma`` down-weights easy,
    already-well-aligned reconstructions and focuses learning on the hard ones.
    """
    recon = F.normalize(recon, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    cos = (recon * target).sum(dim=-1)
    return (1.0 - cos).pow(gamma).mean()


def graphmae_mask(x, mask_rate=0.5, mask_token=None):
    """Mask node rows for GraphMAE and return the reconstruction target.

    Unlike ``mask_atom_features`` (whose target is the 10-way atom type), the
    GraphMAE target is the *full* original feature row of each masked node.

    Returns:
        masked_x:    copy of ``x`` with selected rows overwritten by ``mask_token``.
        mask_index:  (k,) long tensor of masked global row indices.
        target:      (k, in_dim) original feature rows of the masked nodes.
    """
    num_atoms = x.size(0)
    num_mask = max(1, int(num_atoms * mask_rate))
    mask_index = torch.randperm(num_atoms, device=x.device)[:num_mask]

    target = x[mask_index].clone()

    masked_x = x.clone()
    masked_x[mask_index] = mask_token if mask_token is not None else 0.0
    return masked_x, mask_index, target


def augment_view(x, edge_index, edge_attr, mask_token=None,
                 node_mask_rate=0.25, edge_drop_rate=0.25):
    """One stochastic MolCLR view: atom masking + bond (edge) deletion.

    Nodes are preserved (only their features are masked), so the graph→batch
    assignment is unchanged; edges are dropped by column. Returns a fresh
    (x, edge_index, edge_attr) triple.
    """
    x = x.clone()
    n = x.size(0)
    k = int(n * node_mask_rate)
    if k > 0:
        idx = torch.randperm(n, device=x.device)[:k]
        x[idx] = mask_token if mask_token is not None else 0.0

    e = edge_index.size(1)
    if edge_drop_rate > 0 and e > 0:
        keep = torch.rand(e, device=x.device) >= edge_drop_rate
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]
    return x, edge_index, edge_attr


def nt_xent_loss(z1, z2, temperature=0.1) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy (SimCLR / MolCLR).

    z1, z2 are (N, d) projected embeddings of the two views. The positive of each
    anchor is its other-view counterpart; all other 2N−2 graphs are negatives.
    """
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)      # (2N, d)
    sim = torch.matmul(z, z.t()) / temperature             # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))                      # exclude self-similarity
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def mask_atom_features(x, mask_rate=0.15, mask_token=None):
    """Randomly mask atom rows for Stage-1 attribute-masking pretraining.

    Operates on a *batched* node tensor, so masked indices are global and align
    with PyG's concatenated batch (avoids the per-graph offset pitfall).

    Returns:
        masked_x:    copy of ``x`` with selected rows overwritten.
        mask_index:  (k,) long tensor of masked row indices.
        orig_types:  (k,) long tensor of original atom-type ids (argmax of the
                     first NUM_ATOM_TYPES dims) — the reconstruction target.
    """
    num_atoms = x.size(0)
    num_mask = max(1, int(num_atoms * mask_rate))
    mask_index = torch.randperm(num_atoms, device=x.device)[:num_mask]

    orig_types = x[mask_index, :NUM_ATOM_TYPES].argmax(dim=1)

    masked_x = x.clone()
    masked_x[mask_index] = mask_token if mask_token is not None else 0.0
    return masked_x, mask_index, orig_types
