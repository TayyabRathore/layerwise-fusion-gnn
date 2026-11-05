#------------------------------------------------------------------------------#
# data.py -- loading, validating, cleaning and preprocessing the MM-Graph       #
# node-classification datasets (Ele-Fashion and Goodreads-NC).                  #
#                                                                              #
# Pipeline (Fig. 6 of the report, steps 1-4):                                   #
#   1. load the raw benchmark files                                             #
#   2. validate them (splits disjoint, labels valid, features finite, ...)      #
#   3. clean the graph (remove self-loops and duplicate edges, undirected)      #
#   4. preprocess features (L2-normalize the text and image halves separately)  #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Dataset loading, validation, cleaning and preprocessing."""
import os
from typing import Dict, List, NamedTuple

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config

# Feature file of each encoder pair. Every file stores, per node, the text
# embedding followed by the image embedding (text half first).
FEATURE_FILES = {
    "clip": "clip_feat.pt",
    "imagebind": "imagebind_feat.pt",
    "t5vit": "t5vit_feat.pt",
    "t5dino": "t5dino_feat.pt",
}
SPLIT_KEYS = ("train_idx", "val_idx", "test_idx")


class GraphData(NamedTuple):
    """A dataset ready for training.

    Attributes:
        graph: DGL graph on the target device with node data ``feat`` and ``label``.
        splits: node ids of the train / validation / test sets (on the device).
        num_classes: number of classes among labeled nodes.
        text_dim: width of the text half of each feature vector.
    """
    graph: dgl.DGLGraph
    splits: Dict[str, torch.Tensor]
    num_classes: int
    text_dim: int


def _to_numpy(x) -> np.ndarray:
    """Convert a tensor, list or array loaded from a .pt file to a numpy array."""
    if hasattr(x, "numpy"):
        x = x.float() if x.is_floating_point() else x
        x = x.numpy()
    return np.asarray(x)


def load_raw(data_dir: str, name: str, feature: str) -> Dict[str, np.ndarray]:
    """Load the raw benchmark files of one dataset, without any cleaning.

    Args:
        data_dir: folder containing the dataset folders.
        name: dataset folder name, ``"ele-fashion"`` or ``"books-nc"``.
        feature: encoder pair, one of the keys of ``FEATURE_FILES``.

    Returns:
        A dict with ``edges`` (E x 2), ``labels`` (N), the three split arrays
        and ``feat`` (N x D, float32).
    """
    root = os.path.join(data_dir, name)
    edges = _to_numpy(torch.load(os.path.join(root, "nc_edges-nodeid.pt"))).astype(np.int64)
    if edges.shape[0] == 2 and edges.shape[1] != 2:          # accept a 2 x E layout too
        edges = edges.T
    labels = _to_numpy(torch.load(os.path.join(root, "labels-w-missing.pt"))).astype(np.int64)
    split = torch.load(os.path.join(root, "split.pt"))
    feat = _to_numpy(torch.load(os.path.join(root, FEATURE_FILES[feature]), map_location="cpu"))
    raw = {"edges": edges, "labels": labels, "feat": feat.astype(np.float32)}
    for k in SPLIT_KEYS:
        raw[k] = _to_numpy(split[k]).astype(np.int64)
    return raw


def validate(raw: Dict[str, np.ndarray]) -> List[dict]:
    """Run data-quality checks on the raw files (Section III-B of the report).

    Checks that the splits are mutually exclusive and valid, that every split
    node is labeled, that edges point to existing nodes, and that the features
    contain no NaN, infinite or all-zero vectors.

    Args:
        raw: output of :func:`load_raw`.

    Returns:
        A list of dicts with keys ``check``, ``result`` and ``ok`` (True, False,
        or None for purely informative entries).
    """
    n = len(raw["labels"])
    tr, va, te = (raw[k] for k in SPLIT_KEYS)
    rows = []

    def add(check, result, ok=None):
        rows.append({"check": check, "result": result, "ok": ok})

    # splits: disjoint, no repeats, valid indices, labeled
    for (a, b, name) in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
        overlap = len(np.intersect1d(a, b))
        add(f"{name} overlap (nodes)", overlap, overlap == 0)
    repeats = sum(len(s) - len(np.unique(s)) for s in (tr, va, te))
    add("repeated nodes inside a split", repeats, repeats == 0)
    idx = np.concatenate([tr, va, te])
    add("split indices within [0, N)", f"{idx.min()}..{idx.max()} of {n}", idx.min() >= 0 and idx.max() < n)
    add("split ratio train/val/test (%)", "/".join(f"{100 * len(s) / len(idx):.0f}" for s in (tr, va, te)))
    add("nodes outside all splits (not used for training/evaluation)", int(n - len(np.unique(idx))))

    # edges: valid endpoints, self-loops, duplicates
    u, v = raw["edges"][:, 0], raw["edges"][:, 1]
    bad = int(((u < 0) | (u >= n) | (v < 0) | (v >= n)).sum())
    add("edge endpoints outside [0, N)", bad, bad == 0)
    add("edges in file", len(u))
    add("self-loops (removed during cleaning)", int((u == v).sum()))

    # features: shape and finite values
    x = raw["feat"]
    add("feature rows equal number of nodes", f"{x.shape[0]} x {x.shape[1]}", x.shape[0] == n)
    bad_vals = int((~np.isfinite(x)).sum())
    add("NaN or infinite feature values", bad_vals, bad_vals == 0)
    half = x.shape[1] // 2
    zeros = int((np.abs(x[:, :half]).sum(1) == 0).sum() + (np.abs(x[:, half:]).sum(1) == 0).sum())
    add("all-zero text or image vectors", zeros, zeros == 0)
    return rows


def clean_edges(edges: np.ndarray, n: int) -> np.ndarray:
    """Remove self-loops and duplicate edges, and return each undirected edge once.

    Duplicates include an edge stored in both directions, (u, v) and (v, u).

    Args:
        edges: raw edge list, shape (E, 2).
        n: number of nodes.

    Returns:
        Unique undirected edges, shape (E_clean, 2), with u < v.
    """
    u, v = edges[:, 0], edges[:, 1]
    keep = u != v                                           # drop self-loops
    a, b = np.minimum(u[keep], v[keep]), np.maximum(u[keep], v[keep])
    key = np.unique(a * n + b)                              # one id per unordered pair
    return np.stack([key // n, key % n], axis=1)


def normalize_per_modality(feat: np.ndarray, text_dim: int) -> np.ndarray:
    """L2-normalize the text half and the image half of every feature vector separately.

    The EDA (Fig. 4) showed that the two halves differ in scale by up to 100x
    (ImageBind), so without this step one modality would dominate.

    Args:
        feat: features, shape (N, D), text half first.
        text_dim: width of the text half.

    Returns:
        Normalized features with each half of unit length.
    """
    x = torch.from_numpy(feat)
    x = torch.cat([F.normalize(x[:, :text_dim], dim=1), F.normalize(x[:, text_dim:], dim=1)], dim=1)
    return x.numpy()


def load_dataset(cfg: Config, verbose: bool = True) -> GraphData:
    """Load, validate, clean and preprocess one dataset (steps 1-4 of the pipeline).

    Args:
        cfg: configuration; uses ``data_dir``, ``dataset``, ``feature``,
            ``fp16_features`` and ``device``.
        verbose: print a short summary and any failed validation check.

    Returns:
        A :class:`GraphData` with a bidirected graph on ``cfg.device``.

    Raises:
        ValueError: if a validation check fails.
    """
    raw = load_raw(cfg.data_dir, cfg.dataset, cfg.feature)
    n = len(raw["labels"])

    # 2. validation: stop early if the data is broken
    failed = [r for r in validate(raw) if r["ok"] is False]
    if failed:
        raise ValueError(f"Data validation failed: {failed}")

    # 3. cleaning: unique undirected edges, then both directions for message passing
    und = clean_edges(raw["edges"], n)
    src = np.concatenate([und[:, 0], und[:, 1]])
    dst = np.concatenate([und[:, 1], und[:, 0]])
    g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=n)

    # 4. preprocessing: per-modality L2 normalization
    text_dim = raw["feat"].shape[1] // 2
    feat = torch.from_numpy(normalize_per_modality(raw["feat"], text_dim))
    g.ndata["feat"] = feat.half() if cfg.fp16_features else feat
    g.ndata["label"] = torch.from_numpy(raw["labels"])
    g = g.to(cfg.device)

    splits = {k: torch.from_numpy(raw[k]).to(cfg.device) for k in SPLIT_KEYS}
    # classes = labels that occur in the splits (Goodreads-NC uses value 10 for
    # unlabeled books outside the splits; Ele-Fashion never uses class 5)
    used = np.concatenate([raw["labels"][raw[k]] for k in SPLIT_KEYS])
    num_classes = int(used.max()) + 1

    if verbose:
        print(f"{cfg.dataset} [{cfg.feature}]: {n:,} nodes | {len(und):,} undirected edges | "
              f"{len(np.unique(used))} classes present (ids 0..{num_classes - 1}) | "
              f"train/val/test {len(raw['train_idx']):,}/{len(raw['val_idx']):,}/{len(raw['test_idx']):,}")
    return GraphData(g, splits, num_classes, text_dim)
