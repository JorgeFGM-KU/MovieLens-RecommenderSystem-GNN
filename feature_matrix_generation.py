import numpy as np
import torch
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


def get_users_features_matrix(users_file):
    ages = [] # Will be min-max scaled
    genders = [] # Will be one-hot encoded
    occupations = [] # Will be one-hot encoded
    ids = []

    with open(users_file, encoding = "ISO-8859-1") as data_file:
        for line in data_file:
            split = line.split("|")
            id = int(split[0])
            age = int(split[1])
            gender = split[2]
            occupation = split[3]

            ages.append(age)
            genders.append(gender)
            occupations.append(occupation)
            ids.append(id)

    ages = np.array(ages)
    genders = np.array(genders)
    occupations = np.array(occupations)

    minmax_scaler = MinMaxScaler()
    onehot_encoder = OneHotEncoder(sparse=False)

    ages = minmax_scaler.fit_transform(ages.reshape(-1, 1))
    genders = onehot_encoder.fit_transform(genders.reshape(-1, 1))
    occupations = onehot_encoder.fit_transform(occupations.reshape(-1, 1))
    
    features_matrix = np.concatenate((ages, genders, occupations), axis=1, dtype=np.float64)
    return ids, torch.tensor(features_matrix, dtype=torch.float32)

    '''
    unique_zip_codes = dataframe["zip_code"].unique().tolist()
    headers = {"apikey": "cd2d2260-4b37-11ec-bfc5-3156ecbca764"}

    zip_codes_to_query = []
    zip_code_results = {}
    for zc in unique_zip_codes:
        zip_codes_to_query.append(zc)
        if len(zip_codes_to_query) == 100:
            params = {
                "codes": str(zip_codes_to_query)[1:-1].replace("'", ""),
                "country": "US"
            }
            res = requests.get('https://app.zipcodebase.com/api/v1/search', headers=headers, params=params).json()
            for key in res["results"]:
                zip_code_results[key] = res["results"][key]
            zip_codes_to_query = []
    # print(str(dataframe["zip_code"].unique().tolist())[1:-1].replace("'", ""))
    print(len(zip_code_results))
    # print(dataframe)
    '''

def get_items_features_matrix(path_to_raw):
    years = []
    genre_vectors = []
    ids = []

    with open(path_to_raw, encoding = "ISO-8859-1") as data_file:
        for line in data_file:

            split = line.split("|")
            id = int(split[0])
            title = split[1]
            title = title[0:title.find('(')-1]
            release_date = split[2]
            url = split[4]
            
            genre_vector = [int(split[x]) for x in range(5, 23)]
            genre_vector.append(int(split[23][0]))
            
            split_date = release_date.split("-")
            try:
                year = int(split_date[2])
            except: year = 0
            years.append(year)
            genre_vectors.append(genre_vector)
            ids.append(id)

    years = np.array(years)
    genre_vectors = np.array(genre_vectors)

    std_scaler = StandardScaler()
    years = std_scaler.fit_transform(years.reshape(-1,1))

    features_matrix = np.concatenate((years, genre_vectors), axis=1, dtype=np.float64)
    return ids, torch.tensor(features_matrix, dtype=torch.float32)
