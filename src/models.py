#------------------------------------------------------------------------------#
# models.py -- the proposed layer-wise cross-modal fusion GNN and baselines.    #
#                                                                              #
# Model names (used on the command line):                                       #
#   fusion       proposed model: two streams, fused at EVERY layer (Fig. 7)     #
#   fusion-gate  same, with a learned gate instead of the average (ablation)    #
#   late         same two streams, fused only at the output (late fusion)       #
#   sage, gcn, gat   early fusion: one GNN on the concatenated features         #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Model definitions."""
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

MODEL_NAMES = ("fusion", "fusion-gate", "late", "sage", "gcn", "gat")


def _full_neighbor_loader(g, device, batch_size: int):
    """Loader over ALL nodes with their full one-hop neighborhood.

    Used for layer-wise inference: each layer is computed for every node
    using all of its neighbors, without sampling.
    """
    sampler = MultiLayerFullNeighborSampler(1)
    nodes = torch.arange(g.num_nodes(), device=g.device)
    return DataLoader(g, nodes, sampler, device=device, batch_size=batch_size,
                      shuffle=False, drop_last=False, num_workers=0)


class LayerwiseFusionGNN(nn.Module):
    """Two-stream GNN with cross-modal fusion before every message-passing layer.

    Implements Eqs. (1)-(5) of the report:
      (1) each modality is projected to ``hidden_dim`` (inputs are already
          L2-normalized per modality in ``data.py``);
      (2) at layer l the fused vector f = phi(h_text, h_image) is computed;
      (3) f is added to each stream, and each stream aggregates its sampled
          neighbors with its own GraphSAGE layer, followed by
          LayerNorm -> ReLU -> Dropout;
      (5) after the last layer the two streams are concatenated and a linear
          classifier gives the class scores.

    With ``fusion="none"`` no information is exchanged during message passing,
    which gives the late-fusion variant with otherwise identical architecture.
    """

    def __init__(self, in_dim: int, text_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.5, fusion: str = "mean",
                 use_layer_norm: bool = True):
        """Create the model.

        Args:
            in_dim: total feature width (text + image).
            text_dim: width of the text half; the image half is ``in_dim - text_dim``.
            hidden_dim: size d of every stream representation.
            num_classes: number of output classes.
            num_layers: number of message-passing layers L.
            dropout: dropout probability after each layer.
            fusion: ``"mean"`` (average of the streams), ``"gate"`` (learned
                per-dimension gate) or ``"none"`` (late fusion).
            use_layer_norm: apply LayerNorm after every layer.
        """
        super().__init__()
        assert fusion in ("mean", "gate", "none")
        self.text_dim, self.fusion, self.num_layers = text_dim, fusion, num_layers
        self.hidden_dim, self.num_classes = hidden_dim, num_classes

        # Eq. (1): one projection per modality
        self.proj_txt = nn.Linear(text_dim, hidden_dim)
        self.proj_img = nn.Linear(in_dim - text_dim, hidden_dim)
        # Eq. (3): one GraphSAGE layer per stream and per layer
        self.sage_txt = nn.ModuleList(dglnn.SAGEConv(hidden_dim, hidden_dim, "mean") for _ in range(num_layers))
        self.sage_img = nn.ModuleList(dglnn.SAGEConv(hidden_dim, hidden_dim, "mean") for _ in range(num_layers))
        self.norm_txt = nn.ModuleList(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity() for _ in range(num_layers))
        self.norm_img = nn.ModuleList(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity() for _ in range(num_layers))
        # Eq. (2), gated variant: g = sigmoid(W [h_txt || h_img])
        if fusion == "gate":
            self.gates = nn.ModuleList(nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        # Eq. (5): classifier on the concatenated final streams
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def fuse(self, layer: int, h_txt: torch.Tensor, h_img: torch.Tensor) -> torch.Tensor:
        """Eq. (2): the fused vector f of every node at one layer."""
        if self.fusion == "mean":
            return 0.5 * (h_txt + h_img)
        if self.fusion == "gate":
            g = torch.sigmoid(self.gates[layer](torch.cat([h_txt, h_img], dim=1)))
            return g * h_txt + (1.0 - g) * h_img
        return torch.zeros_like(h_txt)                        # "none": late fusion

    def layer(self, l: int, block, h_txt: torch.Tensor, h_img: torch.Tensor):
        """Eq. (3): one fusion + message-passing layer for both streams.

        Args:
            l: layer index.
            block: DGL block whose source nodes are the rows of the inputs.
            h_txt, h_img: stream representations of the block's source nodes.

        Returns:
            The updated representations of the block's destination nodes.
        """
        f = self.fuse(l, h_txt, h_img)
        out_txt = self.sage_txt[l](block, h_txt + f)          # residual injection + aggregation
        out_img = self.sage_img[l](block, h_img + f)
        out_txt = self.dropout(F.relu(self.norm_txt[l](out_txt)))
        out_img = self.dropout(F.relu(self.norm_img[l](out_img)))
        return out_txt, out_img

    def forward(self, blocks, x: torch.Tensor) -> torch.Tensor:
        """Class scores of the target nodes of a sampled mini-batch.

        Args:
            blocks: L sampled message-flow blocks (from DGL's NeighborSampler).
            x: features of the source nodes of the first block.

        Returns:
            Class scores of the destination nodes of the last block.
        """
        x = x.float()                                         # features may be stored as fp16
        h_txt = self.proj_txt(x[:, :self.text_dim])           # Eq. (1)
        h_img = self.proj_img(x[:, self.text_dim:])
        for l, block in enumerate(blocks):
            h_txt, h_img = self.layer(l, block, h_txt, h_img)
        return self.classifier(torch.cat([h_txt, h_img], dim=1))   # Eq. (5)

    @torch.no_grad()
    def inference(self, g, device, batch_size: int) -> torch.Tensor:
        """Class scores of ALL nodes using full neighborhoods, one layer at a time.

        Used for evaluation: every node uses all of its neighbors (no sampling),
        and intermediate results are kept on the CPU to save GPU memory.
        """
        feat, n = g.ndata["feat"], g.num_nodes()
        h_txt = torch.empty(n, self.hidden_dim)
        h_img = torch.empty(n, self.hidden_dim)
        for s in range(0, n, 65536):                          # projection in chunks
            x = feat[s:s + 65536].to(device).float()
            h_txt[s:s + 65536] = self.proj_txt(x[:, :self.text_dim]).cpu()
            h_img[s:s + 65536] = self.proj_img(x[:, self.text_dim:]).cpu()
        loader = _full_neighbor_loader(g, device, batch_size)
        for l in range(self.num_layers):
            new_txt, new_img = torch.empty(n, self.hidden_dim), torch.empty(n, self.hidden_dim)
            t_dev, i_dev = h_txt.to(device), h_img.to(device)
            for input_nodes, output_nodes, blocks in loader:
                o_txt, o_img = self.layer(l, blocks[0], t_dev[input_nodes], i_dev[input_nodes])
                new_txt[output_nodes.cpu()] = o_txt.cpu()
                new_img[output_nodes.cpu()] = o_img.cpu()
            h_txt, h_img = new_txt, new_img
            del t_dev, i_dev
        scores = torch.empty(n, self.num_classes)
        for s in range(0, n, 65536):
            scores[s:s + 65536] = self.classifier(
                torch.cat([h_txt[s:s + 65536], h_img[s:s + 65536]], dim=1).to(device)).cpu()
        return scores


class EarlyFusionGNN(nn.Module):
    """Baseline: one GNN on the concatenated text + image features (early fusion).

    Supports GraphSAGE, GCN and GAT layers with the same depth, width,
    normalization and dropout as the proposed model.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 3,
                 dropout: float = 0.5, conv: str = "sage", use_layer_norm: bool = True, gat_heads: int = 4):
        """Create the baseline.

        Args:
            in_dim: feature width (text + image, concatenated).
            hidden_dim: size of every hidden representation.
            num_classes: number of output classes.
            num_layers: number of message-passing layers.
            dropout: dropout probability after each layer.
            conv: ``"sage"``, ``"gcn"`` or ``"gat"``.
            use_layer_norm: apply LayerNorm after every layer.
            gat_heads: number of attention heads for GAT.
        """
        super().__init__()
        assert conv in ("sage", "gcn", "gat")
        self.conv, self.num_layers, self.hidden_dim, self.num_classes = conv, num_layers, hidden_dim, num_classes
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList()
        for l in range(num_layers):
            if conv == "sage":
                self.convs.append(dglnn.SAGEConv(dims[l], dims[l + 1], "mean"))
            elif conv == "gcn":
                self.convs.append(dglnn.GraphConv(dims[l], dims[l + 1], norm="both", allow_zero_in_degree=True))
            else:   # GAT: heads are concatenated, so each head outputs hidden_dim / heads
                self.convs.append(dglnn.GATConv(dims[l], hidden_dim // gat_heads, num_heads=gat_heads,
                                                allow_zero_in_degree=True))
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity() for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def layer(self, l: int, block, h: torch.Tensor) -> torch.Tensor:
        """One message-passing layer followed by LayerNorm -> ReLU -> Dropout."""
        h = self.convs[l](block, h)
        if self.conv == "gat":
            h = h.flatten(1)                                  # concatenate the heads
        return self.dropout(F.relu(self.norms[l](h)))

    def forward(self, blocks, x: torch.Tensor) -> torch.Tensor:
        """Class scores of the target nodes of a sampled mini-batch."""
        h = x.float()
        for l, block in enumerate(blocks):
            h = self.layer(l, block, h)
        return self.classifier(h)

    @torch.no_grad()
    def inference(self, g, device, batch_size: int) -> torch.Tensor:
        """Class scores of ALL nodes using full neighborhoods, one layer at a time."""
        h, n = g.ndata["feat"], g.num_nodes()
        loader = _full_neighbor_loader(g, device, batch_size)
        for l in range(self.num_layers):
            out = torch.empty(n, self.hidden_dim)
            h_dev = h.to(device)
            for input_nodes, output_nodes, blocks in loader:
                out[output_nodes.cpu()] = self.layer(l, blocks[0], h_dev[input_nodes].float()).cpu()
            h = out
            del h_dev
        scores = torch.empty(n, self.num_classes)
        for s in range(0, n, 65536):
            scores[s:s + 65536] = self.classifier(h[s:s + 65536].to(device)).cpu()
        return scores


def build_model(name: str, in_dim: int, text_dim: int, num_classes: int, cfg) -> nn.Module:
    """Create a model by name with the shared settings from the configuration.

    Args:
        name: one of ``MODEL_NAMES``.
        in_dim: feature width; ``text_dim``: width of its text half.
        num_classes: number of output classes.
        cfg: :class:`src.config.Config`.
    """
    common = dict(hidden_dim=cfg.hidden_dim, num_classes=num_classes, num_layers=cfg.num_layers,
                  dropout=cfg.dropout, use_layer_norm=cfg.use_layer_norm)
    if name in ("fusion", "fusion-gate", "late"):
        fusion = {"fusion": "mean", "fusion-gate": "gate", "late": "none"}[name]
        return LayerwiseFusionGNN(in_dim, text_dim, fusion=fusion, **common)
    if name in ("sage", "gcn", "gat"):
        return EarlyFusionGNN(in_dim, conv=name, gat_heads=cfg.gat_heads, **common)
    raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
