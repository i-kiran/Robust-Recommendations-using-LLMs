import pickle
import json
from data_preprocess import ratings_df
from catboost import datasets, Pool
from catboost.utils import get_roc_curve, eval_metric
# Load user_recommendations from a pickle file
with open('LLM_recommendations.pkl', 'rb') as f:
    responses = pickle.load(f)

# Print the user_recommendations    
print(len(responses))

def parse_responses(responses):
    user_recommendations = {}
    user_recommendations_new = {}
    for response in responses:
        # Parse the JSON response
        data = json.loads(response)
        #print(data)

        # Check if the necessary keys exist in the response
        if 'user_id' in data and 'recommended_items'in data:
            # Extract the user_id and recommended_movies
            user_id = data['user_id']
            recommended_movies = data['recommended_items']

            # Add to the dictionary
            user_recommendations[user_id] = recommended_movies

            # Assign ranks to the recommended movies
            ranked_movies_responses = [(movie_id, len(recommended_movies) - rank + 1) for rank, movie_id in enumerate(recommended_movies, start=1)]

            # Add to the dictionary
            user_recommendations_new[user_id] = ranked_movies_responses
    
    return user_recommendations, user_recommendations_new

def rerank_movies(user_recommendations, ratings_df):
    reranked_recommendations = {}
    reranked_recommendations_with_ranks = {}
    # For each user in user_recommendations
    for user_id, recommended_movies in user_recommendations.items():
        # Initialize an empty list for the user's reranked movies
        reranked_movies = []
        # Loop over all item_ids one by one
        for movie_id in recommended_movies:
            # Extract original rating from ratings_df for that particular user_id and item_id
            rating = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['movie_id'] == movie_id)]['rating'].values

            # If a rating was found, add it to the list
            if len(rating) > 0:
                reranked_movies.append((movie_id, rating[0]))

        # Sort the movies by their rating in descending order
        reranked_movies.sort(key=lambda x: x[1], reverse=True)

        # Add the reranked movies to the reranked_recommendations dictionary
        reranked_recommendations[user_id] = reranked_movies
        # Replace the scores with ranks
        reranked_movies_with_ranks = [(movie_id, len(reranked_movies) - rank + 1) for rank, (movie_id, _) in enumerate(reranked_movies, start=1)]
        # Add the reranked movies to the reranked_recommendations_with_ranks dictionary
        reranked_recommendations_with_ranks[user_id] = reranked_movies_with_ranks

    return reranked_recommendations, reranked_recommendations_with_ranks

def combine_ranks(user_recommendations_new, reranked_recommendations_with_ranks):
    combined_ranks = {}

    # For each user in user_recommendations_new
    for user_id, recommended_movies in user_recommendations_new.items():
        # Initialize an empty dictionary for the user's combined ranks
        combined_ranks_for_user = {}

        # Convert recommended_movies to a dictionary for easy lookup
        recommended_movies_dict = dict(recommended_movies)

        # For each movie and its rank in reranked_recommendations_with_ranks
        for movie_id, reranked_rank in reranked_recommendations_with_ranks[user_id]:
            # If the movie is also in recommended_movies_dict
            if movie_id in recommended_movies_dict:
                # Get the original rank
                rank = recommended_movies_dict[movie_id]

                # Add the movie and its ranks to combined_ranks_for_user
                combined_ranks_for_user[movie_id] = (rank, reranked_rank)

        # Add the combined ranks for the user to combined_ranks
        combined_ranks[user_id] = combined_ranks_for_user

    return combined_ranks

def extract_ranks(combined_ranks):
    original_ranks = {}
    reranked_ranks = {}

    # For each user in combined_ranks
    for user_id, ranks in combined_ranks.items():
        # Initialize empty lists for the user's original and reranked ranks
        original_ranks_for_user = []
        reranked_ranks_for_user = []

        # For each movie and its ranks in ranks
        for movie_id, (original_rank, reranked_rank) in ranks.items():
            # Add the original and reranked ranks to their respective lists
            original_ranks_for_user.append(original_rank)
            reranked_ranks_for_user.append(reranked_rank)

        # Add the lists of ranks for the user to original_ranks and reranked_ranks
        original_ranks[user_id] = original_ranks_for_user
        reranked_ranks[user_id] = reranked_ranks_for_user
        LLM_ranks = original_ranks

    return LLM_ranks, reranked_ranks

def calculate_auc_scores(combined_ranks,LLM_ranks,reranked_ranks):
    auc_scores = {}
    user_ids = list(combined_ranks.keys())
    for user_id in user_ids:
        auc_score = eval_metric(reranked_ranks[user_id],LLM_ranks[user_id], 'AUC:type=Ranking')[0]
        auc_scores[user_id] = auc_score

    sorted_auc_scores = dict(sorted(auc_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_auc_scores
#original_ranks, reranked_ranks = extract_ranks(combine_ranks)

user_recommendations,user_recommendations_new = parse_responses(responses)#generated by llm
reranked_recommendations, reranked_recommendations_with_ranks = rerank_movies(user_recommendations, ratings_df)#generated by user's original ratings
#print(len(user_recommendations))
#print(user_recommendations)
#print("\n\nHOHOHOHOHOHO\n\n")
print(user_recommendations_new)
print("\n\nHOHOHOHOHOHO\n\n")
#print(reranked_recommendations)
#print("\n\nHOHOHOHOHOHO\n\n")
print(reranked_recommendations_with_ranks)
print("\n\nHOHOHOHOHOHO\n\n")
combined_ranks = combine_ranks(user_recommendations_new, reranked_recommendations_with_ranks)
print(combined_ranks)
LLM_ranks, reranked_ranks = extract_ranks(combined_ranks)
print("original_ranks",LLM_ranks)
print("reranked_ranks",reranked_ranks) 
sorted_auc_scores = calculate_auc_scores(combined_ranks,LLM_ranks,reranked_ranks)
print("AUC",sorted_auc_scores)
