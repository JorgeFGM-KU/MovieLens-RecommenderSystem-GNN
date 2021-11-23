import pandas as pd
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocessing import get_data
from load_data import generate_dgl_graph
from feature_matrix_generation import get_users_features_matrix, get_items_features_matrix

df = get_data(raw_data_path="data/raw/")
users = df.users()
movies = df.movies()
ratings = df.ratings()
lens = df.all()
graph = generate_dgl_graph(ratings)
graph.remove_nodes(0, 'user_id')
graph.remove_nodes(0, 'movie_id')
print(graph)

user_ids, user_feat = get_users_features_matrix('data/raw/u.user')
movie_ids, movie_feat = get_items_features_matrix('data/raw/u.item')

print("\n----- user_feat -----\n", user_feat, user_feat.shape)
print("\n----- movie_feat -----\n", movie_feat, movie_feat.shape)
graph.nodes['user_id'].data['feat'] = user_feat
graph.nodes['movie_id'].data['feat'] = movie_feat

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        print("inputs ------------ \n", inputs)
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        #h = self.conv2(graph, h)
        h = self.conv2(graph, h)
        return h

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

k = 5
model = Model(10, 20, 5, graph.etypes)
user_feats = graph.nodes['user_id'].data['feat']
movie_feats = graph.nodes['movie_id'].data['feat']
node_features = {'user_id': user_feats, 'movie_id': movie_feats}
opt = torch.optim.Adam(model.parameters())
for epoch in range(10):
    negative_graph = construct_negative_graph(graph, k, ('user_id', 'rate', 'movie_id'))
    pos_score, neg_score = model(graph, negative_graph, node_features, ('user_id', 'rate', 'movie_id'))
    loss = compute_loss(pos_score, neg_score)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())
