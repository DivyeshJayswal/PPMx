import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.storage import (
    BaseStorage,
    EdgeStorage,
    GlobalStorage,
    NodeStorage,
)
from torch_geometric.nn import GATConv, HeteroConv, Linear, global_mean_pool

torch.serialization.add_safe_globals(
    {
        BaseStorage: BaseStorage,
        NodeStorage: NodeStorage,
        EdgeStorage: EdgeStorage,
        GlobalStorage: GlobalStorage,
    }
)


class InputProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class HeteroGNN(nn.Module):
    """
    PROPHET-style heterogeneous GAT for graph-level process prediction.

    The original paper focuses on next-activity prediction. This repo keeps the
    same heterogeneous graph encoder but exposes three graph-level heads so the
    same backbone can serve classification and regression tasks.
    """

    def __init__(
        self,
        metadata,
        hidden_channels: int,
        proj_dims: dict,
        num_activity_classes: int,
        dropout: float = 0.1,
        loss_weights=(1.0, 1.0, 1.0),
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.loss_weights = loss_weights
        self.node_types = list(node_types)

        self.proj = nn.ModuleDict(
            {
                node_type: InputProjector(proj_dims[node_type], hidden_channels)
                for node_type in node_types
            }
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv(
                    (-1, -1),
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(
                nn.ModuleDict(
                    {node_type: nn.LayerNorm(hidden_channels) for node_type in node_types}
                )
            )

        self.dropout = nn.Dropout(dropout)

        self.out_act = Linear(hidden_channels, num_activity_classes)
        self.out_time = Linear(hidden_channels, 1)
        self.out_rem = Linear(hidden_channels, 1)

    def _coerce_inputs(self, data_or_x_dict, edge_index_dict=None, batch_dict=None):
        if hasattr(data_or_x_dict, "x_dict") and hasattr(data_or_x_dict, "edge_index_dict"):
            data = data_or_x_dict
            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict
            batch_dict = {
                node_type: getattr(data[node_type], "batch", None) for node_type in data.node_types
            }
            return x_dict, edge_index_dict, batch_dict

        if edge_index_dict is None:
            raise ValueError("edge_index_dict is required when passing x_dict directly.")

        if batch_dict is None:
            batch_dict = {}
        return data_or_x_dict, edge_index_dict, batch_dict

    def _pool_graph_embeddings(self, x_dict, batch_dict):
        pooled_sum = None
        pooled_count = None

        for node_type, features in x_dict.items():
            batch = batch_dict.get(node_type)

            if batch is None:
                type_sum = features.sum(dim=0, keepdim=True)
                type_count = torch.tensor(
                    [[features.size(0)]],
                    dtype=features.dtype,
                    device=features.device,
                )
            else:
                type_mean = global_mean_pool(features, batch)
                counts = torch.bincount(batch, minlength=type_mean.size(0)).to(features.device)
                type_sum = type_mean * counts.unsqueeze(-1)
                type_count = counts.unsqueeze(-1).to(features.dtype)

            if pooled_sum is None:
                pooled_sum = type_sum
                pooled_count = type_count
            else:
                pooled_sum = pooled_sum + type_sum
                pooled_count = pooled_count + type_count

        return pooled_sum / pooled_count.clamp_min(1.0)

    def forward(self, data_or_x_dict, edge_index_dict=None, batch_dict=None):
        x_dict, edge_index_dict, batch_dict = self._coerce_inputs(
            data_or_x_dict,
            edge_index_dict=edge_index_dict,
            batch_dict=batch_dict,
        )

        hidden_dict = {node_type: self.proj[node_type](x) for node_type, x in x_dict.items()}

        for conv, norm_dict in zip(self.convs, self.norms):
            updated = conv(hidden_dict, edge_index_dict)
            next_hidden = {}
            for node_type in hidden_dict:
                node_update = updated.get(node_type, hidden_dict[node_type])
                node_update = norm_dict[node_type](node_update)
                node_update = F.elu(node_update)
                next_hidden[node_type] = self.dropout(node_update)
            hidden_dict = next_hidden

        graph_embedding = self._pool_graph_embeddings(hidden_dict, batch_dict)
        act = self.out_act(graph_embedding)
        time = self.out_time(graph_embedding).squeeze(-1)
        rem = self.out_rem(graph_embedding).squeeze(-1)
        return act, time, rem

    def compute_loss(self, act_logits, time_pred, rem_pred, batch):
        y_act = batch.y_activity.view(-1)
        y_time = batch.y_timestamp.view(-1)
        y_rem = batch.y_remaining_time.view(-1)

        w_act, w_time, w_rem = self.loss_weights
        loss = (
            w_act * F.cross_entropy(act_logits, y_act)
            + w_time * F.l1_loss(time_pred, y_time)
            + w_rem * F.l1_loss(rem_pred, y_rem)
        )
        return loss
