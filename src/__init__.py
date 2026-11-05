#------------------------------------------------------------------------------#
# Layer-wise Cross-Modal Fusion in Graph Neural Networks                        #
# for Multimodal Node Classification                                            #
#                                                                              #
# ESOF-5052 Big Data Machine Learning, Lakehead University, Fall 2026           #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Package for the project. Each module handles one stage of the pipeline:

config    -- all hyperparameters in one place (matches the report's table)
data      -- loading, validation, cleaning and per-modality normalization
augment   -- training-time augmentation (modality dropout, edge dropout)
models    -- the proposed layer-wise fusion GNN and the baseline GNNs
train     -- training loop with best-validation model selection
evaluate  -- test metrics, inference time, model size, result storage
plots     -- figures for the report
"""
__version__ = "0.5.0"
