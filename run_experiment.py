#------------------------------------------------------------------------------#
# run_experiment.py -- train and evaluate models from the command line.         #
#                                                                              #
# Example (proposed model + GraphSAGE baseline, 3 seeds, Ele-Fashion, CLIP):    #
#   python run_experiment.py --dataset ele-fashion --feature clip \             #
#          --models fusion sage --seeds 0 1 2                                   #
# Quick check that everything works (1 seed, 2 epochs):                         #
#   python run_experiment.py --quick                                            #
#                                                                              #
# Finished runs are stored in results/results.json and skipped if the same      #
# command is run again, so an interrupted experiment can simply be restarted.   #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Command-line entry point for training and evaluation."""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data import load_dataset
from src.evaluate import evaluate_on_test, load_results, run_key, save_result
from src.models import MODEL_NAMES, build_model, count_parameters
from src.plots import plot_model_comparison, plot_training_curves
from src.train import train_model


def parse_args() -> argparse.Namespace:
    """Read the command-line options; anything not given keeps its value from Config."""
    d = Config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=d.dataset, choices=["ele-fashion", "books-nc"])
    p.add_argument("--feature", default=d.feature, choices=["clip", "imagebind", "t5vit", "t5dino"])
    p.add_argument("--models", nargs="+", default=["fusion", "sage"], choices=MODEL_NAMES)
    p.add_argument("--seeds", nargs="+", type=int, default=d.seeds)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--num-neighbors", type=int, default=d.num_neighbors)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--hidden-dim", type=int, default=d.hidden_dim)
    p.add_argument("--modality-dropout", type=float, default=d.modality_dropout)
    p.add_argument("--edge-dropout", type=float, default=d.edge_dropout)
    p.add_argument("--class-weighted-loss", action="store_true")
    p.add_argument("--no-layer-norm", action="store_true")
    p.add_argument("--fp16", action="store_true", help="store features in half precision (large datasets)")
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--results-dir", default=d.results_dir)
    p.add_argument("--figures-dir", default=d.figures_dir)
    p.add_argument("--quick", action="store_true", help="smoke test: 1 seed, 2 epochs, separate results file")
    return p.parse_args()


def make_config(a: argparse.Namespace) -> Config:
    """Build the Config for this run from the command-line options."""
    cfg = Config(dataset=a.dataset, feature=a.feature, epochs=a.epochs, num_neighbors=a.num_neighbors,
                 batch_size=a.batch_size, hidden_dim=a.hidden_dim, modality_dropout=a.modality_dropout,
                 edge_dropout=a.edge_dropout, class_weighted_loss=a.class_weighted_loss,
                 use_layer_norm=not a.no_layer_norm, fp16_features=a.fp16, data_dir=a.data_dir,
                 results_dir=a.results_dir, figures_dir=a.figures_dir, seeds=a.seeds,
                 device="cuda" if torch.cuda.is_available() else "cpu")
    if a.quick:
        cfg.epochs, cfg.seeds = 2, [0]
    return cfg


def summarize(results: dict, cfg: Config, models) -> pd.DataFrame:
    """Mean and standard deviation over seeds of every reported quantity, per model."""
    rows = []
    for m in models:
        runs = [v for k, v in results.items() if k.startswith(f"{cfg.dataset}|{cfg.feature}|{m}|")]
        if not runs:
            continue
        row = {"model": m, "runs": len(runs)}
        for key in ("accuracy", "macro_f1", "epoch_time_s", "inference_us_per_node", "peak_memory_gb"):
            vals = np.array([r[key] for r in runs])
            row[f"{key}_mean"] = vals.mean()
            row[f"{key}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        row["parameters"] = runs[0]["parameters"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Train and evaluate every requested model and seed, then summarize and plot."""
    a = parse_args()
    cfg = make_config(a)
    results_file = os.path.join(cfg.results_dir, "results_quick.json" if a.quick else "results.json")
    print(f"device: {cfg.device} | results: {results_file}")

    data = load_dataset(cfg)                                   # steps 1-4: load, validate, clean, normalize
    in_dim = data.graph.ndata["feat"].shape[1]

    for model_name in a.models:
        for seed in cfg.seeds:
            key = run_key(cfg, model_name, seed)
            if key in load_results(results_file):
                print(f"[skip] {key} (already done)")
                continue
            print(f"[run ] {key}")
            torch.manual_seed(seed)                            # identical initial weights per seed
            model = build_model(model_name, in_dim, data.text_dim, data.num_classes, cfg).to(cfg.device)
            trained = train_model(model, data, cfg, seed)
            model.load_state_dict(trained["best_state"])       # best-validation weights
            test = evaluate_on_test(model, data, cfg)          # test set used once
            result = {
                "accuracy": test["accuracy"], "macro_f1": test["macro_f1"],
                "per_class_f1": test["per_class_f1"], "confusion_matrix": test["confusion_matrix"],
                "classes": test["classes"], "val_accuracy_best": trained["best_val_acc"],
                "best_epoch": trained["best_epoch"], "parameters": count_parameters(model),
                "epoch_time_s": float(np.mean([h["epoch_time"] for h in trained["history"]])),
                "inference_time_s": test["inference_time_s"], "inference_us_per_node": test["inference_us_per_node"],
                "peak_memory_gb": trained["peak_memory_gb"], "history": trained["history"],
                "config": {k: v for k, v in vars(cfg).items() if k != "seeds"},
            }
            save_result(results_file, key, result)
            print(f"       test accuracy {test['accuracy']:.4f} | macro-F1 {test['macro_f1']:.4f} | "
                  f"best epoch {trained['best_epoch']} | {result['parameters']:,} parameters")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # summary table and figures
    results = load_results(results_file)
    summary = summarize(results, cfg, a.models)
    if len(summary):
        tag = f"{cfg.dataset}_{cfg.feature}" + ("_quick" if a.quick else "")
        summary.to_csv(os.path.join(cfg.results_dir, f"summary_{tag}.csv"), index=False)
        print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        histories = {m: [v["history"] for k, v in results.items() if k.startswith(f"{cfg.dataset}|{cfg.feature}|{m}|")]
                     for m in summary["model"]}
        plot_training_curves(histories, f"{cfg.dataset} ({cfg.feature})", os.path.join(cfg.figures_dir, f"curves_{tag}"))
        plot_model_comparison(summary, "accuracy", os.path.join(cfg.figures_dir, f"accuracy_{tag}"))
        plot_model_comparison(summary, "macro_f1", os.path.join(cfg.figures_dir, f"macro_f1_{tag}"))


if __name__ == "__main__":
    main()
