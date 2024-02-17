from catboost import datasets, Pool
from catboost.utils import get_roc_curve, eval_metric
from train_model import ratings_df,trainset, testset, predictions
import pandas as pd

def convert_to_dataframe(testset):
    test_df = pd.DataFrame(testset, columns=['user_id', 'movie_id', 'rating'])
    return test_df

def count_user_ratings(test_df):
    user_rating_counts = test_df['user_id'].value_counts()
    return user_rating_counts

def find_min_max_ratings(user_rating_counts):
    min_ratings = user_rating_counts.min()
    max_ratings = user_rating_counts.max()
    return min_ratings, max_ratings

def filter_users_with_less_than_5_ratings(test_df, user_rating_counts):
    users_with_less_than_5_ratings = user_rating_counts[user_rating_counts < 5].index
    test_df = test_df[~test_df['user_id'].isin(users_with_less_than_5_ratings)]
    return test_df, users_with_less_than_5_ratings

def get_user_ratings(predictions, test_df):
    user_ratings = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid in test_df['user_id'].values:
            if uid in user_ratings:
                user_ratings[uid][0].append(true_r)
                user_ratings[uid][1].append(est)
            else:
                user_ratings[uid] = ([true_r], [est])
    return user_ratings

def calculate_auc_scores(user_ratings):
    auc_scores = {}
    user_ids = list(user_ratings.keys())
    for user_id in user_ids:
        auc_score = eval_metric(user_ratings[user_id][0],user_ratings[user_id][1], 'AUC:type=Ranking')[0]
        auc_scores[user_id] = auc_score

    sorted_auc_scores = dict(sorted(auc_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_auc_scores

def calculate_sparsity_scores(ratings_df, test_df):
    pivot_table = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')
    sparsity_values = {}
    # Iterate over the users in the new test set
    for user_id in test_df['user_id'].unique():
        user_ratings = pivot_table.loc[user_id]
        sparsity = user_ratings.isnull().sum() / len(user_ratings)
        sparsity_values[user_id] = sparsity

    sorted_sparsity_scores = dict(sorted(sparsity_values.items(), key=lambda item: item[1], reverse=True))
    return sorted_sparsity_scores


def calculate_metrics(testset, predictions, ratings_df):
    test_df = convert_to_dataframe(testset)
    user_rating_counts = count_user_ratings(test_df)
    min_ratings, max_ratings = find_min_max_ratings(user_rating_counts)
    test_df, users_with_less_than_5_ratings = filter_users_with_less_than_5_ratings(test_df, user_rating_counts)
    user_ratings = get_user_ratings(predictions, test_df)
    sorted_auc_scores = calculate_auc_scores(user_ratings)
    sorted_sparsity_scores = calculate_sparsity_scores(ratings_df, test_df)
    return sorted_auc_scores, sorted_sparsity_scores