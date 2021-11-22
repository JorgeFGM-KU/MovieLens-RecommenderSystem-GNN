from feature_matrix_generation import get_items_features_matrix, get_users_features_matrix

# Note that NOT ALL THE USERS/MOVIES may be in the matrixes (some of tehn have been discarded
# because data was missing), so make sure to check user_ids and movies_ids first to make sure!

user_ids, users_features_mat = get_users_features_matrix("data/raw/u.user")
movies_ids, movies_features_mat = get_items_features_matrix("data/raw/u.item")


print(f"Number of users: {users_features_mat.shape[0]} - Number of features per user: {users_features_mat.shape[1]}")
print(users_features_mat)
print(f"Number of movies: {movies_features_mat.shape[0]} - Number of features per movie: {movies_features_mat.shape[1]}")
print(movies_features_mat)