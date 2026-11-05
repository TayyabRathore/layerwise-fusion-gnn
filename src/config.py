#------------------------------------------------------------------------------#
# config.py -- all hyperparameters of the project in one place.                 #
# The default values are the training setup reported in Section IV-F of the     #
# report, and they are identical for every model so comparisons are fair.       #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Experiment configuration."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """All settings of one experiment.

    Attributes are grouped by purpose. Change values here (or through the
    command-line flags of ``run_experiment.py``) instead of inside the code.
    """

    # ---- paths and hardware
    data_dir: str = "data"              # folder that contains ele-fashion/ and books-nc/
    results_dir: str = "results"        # JSON results, logs and per-run histories
    figures_dir: str = "figures"        # figures for the report
    device: str = "cuda"                # "cuda" or "cpu"

    # ---- data
    dataset: str = "ele-fashion"        # "ele-fashion" or "books-nc"
    feature: str = "clip"               # "clip", "imagebind", "t5vit" or "t5dino"
    fp16_features: bool = False         # store features in half precision (saves GPU memory)

    # ---- model
    hidden_dim: int = 128               # d: size of every hidden representation
    num_layers: int = 3                 # L: number of message-passing layers
    dropout: float = 0.5                # dropout after every hidden layer
    use_layer_norm: bool = True         # LayerNorm after every layer (all models)
    gat_heads: int = 4                  # attention heads in the GAT baseline

    # ---- neighbor sampling and batching
    num_neighbors: int = 15             # neighbors sampled per layer during training
    batch_size: int = 128               # target nodes per mini-batch
    infer_batch_size: int = 1024        # nodes per batch in full-graph inference

    # ---- optimization
    epochs: int = 30
    lr: float = 1e-3                    # initial learning rate (cosine decay to 0)
    weight_decay: float = 5e-4

    # ---- augmentation (training only)
    modality_dropout: float = 0.15      # share of nodes per batch with text or image hidden
    edge_dropout: float = 0.15          # share of edges hidden from the sampler each epoch

    # ---- class imbalance
    class_weighted_loss: bool = False   # weight the loss by inverse class frequency

    # ---- evaluation protocol
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
