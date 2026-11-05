#------------------------------------------------------------------------------#
# train.py -- training with neighbor sampling and best-validation selection.    #
#                                                                              #
# Protocol (Section IV-F): train for a fixed number of epochs; after every      #
# epoch measure accuracy on the VALIDATION set only; keep the weights of the    #
# best validation epoch. The test set is never used here.                       #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Model training."""
import copy
import random
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.dataloading import DataLoader, NeighborSampler

from .augment import edge_keep_prob, modality_dropout
from .config import Config
from .data import GraphData


def set_seed(seed: int) -> None:
    """Fix every random number generator, so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def make_loader(g, node_ids: torch.Tensor, cfg: Config, shuffle: bool, edge_prob: str = None) -> DataLoader:
    """Mini-batch loader that samples ``cfg.num_neighbors`` neighbors per layer.

    Args:
        g: the graph.
        node_ids: target nodes (e.g. the training nodes).
        cfg: configuration (batch size, neighbors, layers, device).
        shuffle: shuffle the target nodes every epoch (training only).
        edge_prob: name of an edge feature with sampling probabilities; edges
            with probability 0 are never sampled (used for edge dropout).
            ``None`` samples every edge uniformly.
    """
    fanouts = [cfg.num_neighbors] * cfg.num_layers
    if edge_prob is None:
        sampler = NeighborSampler(fanouts, prefetch_node_feats=["feat"])
    else:
        sampler = NeighborSampler(fanouts, prefetch_node_feats=["feat"], prob=edge_prob)
    return DataLoader(g, node_ids, sampler, device=cfg.device, batch_size=cfg.batch_size,
                      shuffle=shuffle, drop_last=False, num_workers=0)


def class_weights(labels: torch.Tensor, train_idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights w_c = N / (C * n_c) from the training labels.

    Rare classes get weights above 1, frequent classes below 1. Classes absent
    from the training set get weight 0.
    """
    counts = torch.bincount(labels[train_idx], minlength=num_classes).float()
    present = counts > 0
    w = torch.zeros(num_classes, device=labels.device)
    w[present] = counts.sum() / (present.sum() * counts[present])
    return w


@torch.no_grad()
def evaluate_sampled(model, loader) -> float:
    """Accuracy on a sampled loader (used for validation during training)."""
    model.eval()
    correct = total = 0
    for _, _, blocks in loader:
        pred = model(blocks, blocks[0].srcdata["feat"]).argmax(1)
        y = blocks[-1].dstdata["label"]
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_model(model, data: GraphData, cfg: Config, seed: int, verbose: bool = True) -> dict:
    """Train a model and return the best-validation weights and the training history.

    Each epoch: draw a new edge-dropout mask, train on all training nodes in
    mini-batches with modality dropout, then measure validation accuracy.
    The learning rate follows a cosine schedule from ``cfg.lr`` to 0.

    Args:
        model: a model from :mod:`src.models` (already on ``cfg.device``).
        data: the loaded dataset.
        cfg: configuration.
        seed: random seed of this run.
        verbose: print one line per epoch.

    Returns:
        A dict with ``best_state`` (weights), ``best_epoch``, ``best_val_acc``,
        ``history`` (per-epoch loss, validation accuracy and time) and
        ``peak_memory_gb``.
    """
    set_seed(seed)
    g, splits = data.graph, data.splits

    # edge dropout: the sampler never picks edges whose "keep" probability is 0
    use_edge_dropout = cfg.edge_dropout > 0
    if use_edge_dropout:
        g.edata["keep"] = edge_keep_prob(g.num_edges(), cfg.edge_dropout, g.device)
        try:
            train_loader = make_loader(g, splits["train_idx"], cfg, shuffle=True, edge_prob="keep")
            next(iter(train_loader))                         # test once that this DGL build supports it
        except Exception as err:                             # e.g. no probability sampling on this device
            print(f"  warning: edge dropout disabled, the sampler does not support it here ({type(err).__name__})")
            use_edge_dropout = False
            del g.edata["keep"]
    if not use_edge_dropout:
        train_loader = make_loader(g, splits["train_idx"], cfg, shuffle=True)
    val_loader = make_loader(g, splits["val_idx"], cfg, shuffle=False)   # validation sees every edge

    weight = class_weights(g.ndata["label"], splits["train_idx"], data.num_classes) \
        if cfg.class_weighted_loss else None
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    history, best = [], {"val": -1.0, "epoch": -1, "state": None}
    for epoch in range(cfg.epochs):
        if use_edge_dropout:                                 # new random graph view every epoch
            g.edata["keep"] = edge_keep_prob(g.num_edges(), cfg.edge_dropout, g.device)
        model.train()
        t0, total_loss, steps = time.time(), 0.0, 0
        for _, _, blocks in train_loader:
            x = modality_dropout(blocks[0].srcdata["feat"].float(), data.text_dim, cfg.modality_dropout)
            y = blocks[-1].dstdata["label"]
            loss = F.cross_entropy(model(blocks, x), y, weight=weight)    # Eq. (6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - t0
        val_acc = evaluate_sampled(model, val_loader)
        sched.step()

        history.append({"epoch": epoch, "loss": total_loss / steps, "val_acc": val_acc, "epoch_time": epoch_time})
        if val_acc > best["val"]:                            # model selection on validation only
            best = {"val": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if verbose:
            print(f"  epoch {epoch:02d} | loss {total_loss / steps:.4f} | val acc {val_acc:.4f} | {epoch_time:.1f}s")

    if use_edge_dropout:
        del g.edata["keep"]
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return {"best_state": best["state"], "best_epoch": best["epoch"], "best_val_acc": best["val"],
            "history": history, "peak_memory_gb": peak}
