from feature_matrix_generation import get_items_features_matrix, get_users_features_matrix


users_features_mat = get_users_features_matrix("data/raw/u.user")
movies_features_mat = get_items_features_matrix("data/raw/u.item")

print(users_features_mat)
print(movies_features_mat)