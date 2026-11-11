"""Graph neural network for whole-neuron classification."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class NeuronSAGE(torch.nn.Module):
    """Three GraphSAGE layers, mean pooling over nodes, then a linear classifier.

    GraphSAGE: Hamilton, Ying & Leskovec (2017), NeurIPS.
    """

    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [SAGEConv(in_dim, hidden), SAGEConv(hidden, hidden), SAGEConv(hidden, hidden)]
        )
        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)          # one vector per neuron
        x = F.dropout(x, self.dropout, training=self.training)
        return self.lin(x)
