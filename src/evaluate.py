#------------------------------------------------------------------------------#
# evaluate.py -- final evaluation on the test set and result storage.           #
#                                                                              #
# The test set is used ONCE per run, with the best-validation weights.          #
# Reported: accuracy, macro-F1 (all classes count equally, Section III-G),      #
# per-class F1, inference time (total and per node), parameters, GPU memory.    #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Test evaluation, metrics and result files."""
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import GraphData


@torch.no_grad()
def predict(model, data: GraphData, device, batch_size: int):
    """Full-graph predictions for every node, and the time they took.

    Returns:
        (scores of all nodes as a CPU tensor, inference time in seconds)
    """
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    scores = model.inference(data.graph, device, batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return scores, time.time() - t0


def classification_metrics(scores: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> dict:
    """Accuracy, macro-F1, per-class F1 and confusion matrix on the nodes ``idx``.

    Macro-F1 is computed over the classes that occur in ``idx`` (Ele-Fashion's
    unused class 5 therefore does not lower it).
    """
    idx = idx.cpu()
    y_true = labels.cpu()[idx].numpy()
    y_pred = scores[idx].argmax(1).numpy()
    classes = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "per_class_f1": {int(c): float(f)
                         for c, f in zip(classes, f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classes": [int(c) for c in classes],
    }


def evaluate_on_test(model, data: GraphData, cfg) -> dict:
    """Run the final test evaluation of a trained model (best-validation weights loaded).

    Returns:
        Test metrics plus ``inference_time_s`` (whole graph) and
        ``inference_us_per_node`` (microseconds per node).
    """
    scores, seconds = predict(model, data, cfg.device, cfg.infer_batch_size)
    out = classification_metrics(scores, data.graph.ndata["label"], data.splits["test_idx"])
    out["val_accuracy_full"] = classification_metrics(scores, data.graph.ndata["label"], data.splits["val_idx"])["accuracy"]
    out["inference_time_s"] = seconds
    out["inference_us_per_node"] = 1e6 * seconds / data.graph.num_nodes()
    return out


# ---------------------------------------------------------------- result storage
def run_key(cfg, model_name: str, seed: int) -> str:
    """Unique name of one run, e.g. ``ele-fashion|clip|fusion|seed0``."""
    return f"{cfg.dataset}|{cfg.feature}|{model_name}|seed{seed}"


def load_results(path: str) -> dict:
    """Read all saved runs (empty dict if the file does not exist yet)."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_result(path: str, key: str, value: dict) -> None:
    """Add one run to the results file.

    The file is written to a temporary name first and then renamed, so a crash
    can never leave a half-written results file.
    """
    results = load_results(path)
    results[key] = value
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, path)
