import dgl
import torch
from dgl.nn import SAGEConv
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from data_preprocessing import get_data
from feature_matrix_generation import get_users_features_matrix, get_items_features_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ratings = get_data(raw_data_path="data/raw").ratings()
users_ids, users_feature_matrix = get_users_features_matrix("data/raw/u.user")
movies_ids, movies_feature_matrix = get_items_features_matrix("data/raw/u.item")

graph = dgl.heterograph(
    {("user", "rating", "movie"): (
        torch.tensor(ratings["user_id"].to_numpy(), dtype=torch.int32),
        torch.tensor(ratings["movie_id"].to_numpy(), dtype=torch.int32)
    )},
    device=device
)
graph.remove_nodes(0, 'user')
graph.remove_nodes(0, 'movie')
graph.edges["rating"].data['w'] = torch.tensor(ratings["rating"].to_numpy(), dtype=torch.float32)
graph.nodes["user"].data["feature_vector"] = users_feature_matrix
graph.nodes["movie"].data["feature_vector"] = movies_feature_matrix

print(users_feature_matrix.shape)
print(movies_feature_matrix.shape)
print(graph.ndata["user"])


class SAGE(nn.Module):
    def __init__(self, in_feats=(24,20), hid_feats=64, out_feats=128):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

idk = SAGE()
idk(graph, (graph.nodes["user"].data["feature_vector"], graph.nodes["movie"].data["feature_vector"]))