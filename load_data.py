import pandas as pd
import dgl
import torch

def generate_dgl_graph(data: pd.DataFrame):
    # We generate a graph with "user_id" nodes as source, edge types "rating" and
    # "movie_id" nodes as destinations.
    # We then add the "rating" values as weights of edges.
    # + We may add some lines about embeddings of nodes
    # after the embedding vector task is done.

    # Our graph has 2 node types (user type, movie type), so it is heterogeneous graph
    graph_data = {
        ('user_id', 'rate', 'movie_id') : (data["user_id"], data["movie_id"]),
        ('movie_id', 'rated', 'user_id') : (data["movie_id"], data["user_id"])
    }
    dgl_graph = dgl.heterograph(graph_data)
    dgl_graph.edges["rate"].data["score"] = torch.tensor(data["rating"])
    dgl_graph.edges["rated"].data["score"] = torch.tensor(data["rating"])
    return dgl_graph