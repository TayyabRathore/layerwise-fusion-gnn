#------------------------------------------------------------------------------#
# augment.py -- training-time data augmentation (Section IV-E of the report).   #
# Both techniques are applied ONLY during training, never to validation or     #
# test data.                                                                    #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Modality dropout and edge dropout."""
import torch


def modality_dropout(x: torch.Tensor, text_dim: int, p: float) -> torch.Tensor:
    """Hide the text OR the image half of a random share of nodes.

    For each node, with probability ``p`` one modality (chosen at random,
    never both) is replaced by zeros. This forces the model to learn from both
    modalities instead of relying on the stronger one (EDA, Table IV).

    Args:
        x: input features of the nodes in a mini-batch, shape (B, D).
        text_dim: width of the text half; the image half is the rest.
        p: probability that a node loses one modality.

    Returns:
        A new tensor with the dropped halves set to zero (``x`` is unchanged).
    """
    if p <= 0:
        return x
    n = x.shape[0]
    drop = torch.rand(n, device=x.device) < p             # nodes that lose a modality
    drop_text = torch.rand(n, device=x.device) < 0.5      # which modality they lose
    x = x.clone()
    x[drop & drop_text, :text_dim] = 0.0                  # hide the text half
    x[drop & ~drop_text, text_dim:] = 0.0                 # hide the image half
    return x


def edge_keep_prob(num_edges: int, p: float, device) -> torch.Tensor:
    """Per-edge sampling probability for edge dropout: 1.0 = kept, 0.0 = dropped.

    Each edge is dropped with probability ``p``. The values are stored as an edge
    feature and passed to the neighbor sampler as sampling probabilities, so a
    dropped edge (probability 0) is never sampled. A new draw is made every epoch,
    so the model sees a slightly different graph each time (edge dropout).
    (DGL cannot sample with a boolean edge mask on the GPU, but it can sample
    with float probabilities, which has the same effect here.)

    Args:
        num_edges: number of (directed) edges in the graph.
        p: probability of dropping an edge.
        device: device of the graph.

    Returns:
        Float tensor of shape (num_edges,) with values 0.0 or 1.0.
    """
    return (torch.rand(num_edges, device=device) >= p).float()
