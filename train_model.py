from data_preprocess import load_data
import numpy as np
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader, Dataset


# Load the data
ratings_df, users_df, items_df = load_data()
def load_dataset(ratings_df):
    reader = Reader(rating_scale=(1, 5))  # Rating scale is from 1 to 5
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
    return data

def cross_validation(svd, data):
    results_SVD = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
    print("Average MAE: ", np.average(results_SVD["test_mae"]))
    print("Average RMSE: ", np.average(results_SVD["test_rmse"]))


def grid_search(data):
    param_grid = {'n_factors': [20, 50, 100], 
                  'n_epochs': [5, 10, 20]}
    Grid_Search = GridSearchCV(SVD, param_grid, measures = ['rmse', 'mae'], cv = 10)
    Grid_Search.fit(data)
    print(Grid_Search.best_score['rmse'])
    print(Grid_Search.best_params['rmse'])
    return Grid_Search.best_params['rmse']

def train_and_test(data, best_factor, best_epoch):
    trainset, testset = train_test_split(data, test_size = .20)
    svd = SVD(n_factors = best_factor, n_epochs = best_epoch)
    svd.fit(trainset)
    predictions = svd.test(testset)
    return trainset, testset, predictions

# Create an SVD algorithm object
svd = SVD()

# Load the dataset
data = load_dataset(ratings_df)

# Perform cross validation
cross_validation(svd, data)

# Perform grid search
best_params = grid_search(data)

# Train and test
trainset, testset, predictions = train_and_test(data, best_params['n_factors'], best_params['n_epochs'])
