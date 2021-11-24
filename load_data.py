import pandas as pd
import dgl
import torch
import stellargraph as sg
from data_preprocessing import get_data
from feature_matrix_generation import get_users_features_matrix, get_items_features_matrix
from sklearn import preprocessing

def generate_stellar_graph():
    df = get_data()
    users = df.users()
    movies = df.movies()
    ratings = df.ratings()
    movies = movies[movies["movie_id"] != 267]
    ratings = ratings[ratings["movie_id"] != 267]

    users_ids = "user_" + users["user_id"].astype(str)
    movies["movie_id"] = "movie_" + movies["movie_id"].astype(str)
    movies.set_index("movie_id", inplace = True)
    ratings["user_id"] = "user_" + ratings["user_id"].astype(str)
    ratings["movie_id"] = "movie_" + ratings["movie_id"].astype(str)

    ########## add encoded features to user and movie data

    user_feat = get_users_features_matrix('data/raw/u.user')
    movie_feat = get_items_features_matrix('data/raw/u.item')
    encoded_movies = movies.assign(release_date = movie_feat[:,0])

    feature_encoding = preprocessing.OneHotEncoder(sparse = False)
    onehot = feature_encoding.fit_transform(users[["sex", "occupation"]])
    scaled_age = preprocessing.scale(users["age"])
    encoded_users = pd.DataFrame(onehot, index = users_ids).assign(scaled_age = scaled_age)

    #print("\n\n-------- encoded_users --------\n", encoded_users)
    #print("\n\n-------- encoded_movies --------\n", encoded_movies)
    #print("\n\n-------- ratings --------\n", ratings)
    g = sg.StellarGraph(
        {"user" : encoded_users, "movie" : encoded_movies},
        {"rating" : ratings[["user_id", "movie_id"]]},
        source_column = "user_id",
        target_column = "movie_id",
    )
    return g, ratings


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
