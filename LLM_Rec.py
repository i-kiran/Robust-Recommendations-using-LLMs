from prompt_generation import prompts, df_items
import openai
import pandas as pd
import random
import json
from data_preprocess import items_df
import pickle
def generate_responses(prompts):
    # Initialize a list to store the responses
    responses = []
    openai.api_key = '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'
    # Iterate over the prompts
    for prompt in prompts:
        # Generate a response to the prompt
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            temperature = 0,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a very smart user preference based recommendation system designed to output JSON."},
                {"role": "user", "content": prompt}
            ]
            
        )
        
        # Append the response to the list
        responses.append(response.choices[0].message.content)
    
    return responses

'''
def parse_responses(responses):
    # Assuming responses is a list of JSON strings
    user_recommendations = {}

    for response in responses:
        # Parse the JSON response
        data = json.loads(response)

        # Check if the necessary keys exist in the response
        if 'user_id' in data and 'recommended_movies' in data:
            # Extract the user_id and recommended_movies
            user_id = data['user_id']
            recommended_movies = data['recommended_movies']

            # Add to the dictionary
            user_recommendations[user_id] = recommended_movies
    
    return user_recommendations


def map_titles_to_ids(items_df, user_recommendations, df_items):
    # Create a dictionary mapping movie titles to movie_ids
    movie_to_id = pd.Series(items_df.movie_id.values,index=items_df.movie_title).to_dict()
    )
    for user_id, movies in user_recommendations.items():
        # Replace movie names with movie_ids
        item_ids = [movie_to_id.get(movie) for movie in movies if movie_to_id.get(movie) is not None]
        user_recommendations[user_id] = item_ids

    # Create a dictionary mapping movie titles to movie_ids
    title_to_item_id = items_df.set_index('movie_title')['movie_id'].to_dict()

    # Initialize an empty dictionary
    test_items_dict = {}

    # Iterate over each row in the DataFrame
    for index, row in df_items.iterrows():
        # Flatten the list of lists into a single list
        flat_list = [item for sublist in row['test_items'] for item in sublist]
        # Replace movie titles with movie_ids in the list
        item_ids = [title_to_item_id.get(item) for item in flat_list if title_to_item_id.get(item) is not None]
        # Add the user_id and item_ids to the dictionary
        test_items_dict[row['user_id']] = item_ids
    
    return user_recommendations, test_items_dict
'''
responses = generate_responses(prompts)
print(responses)
#user_recommendations = parse_responses(responses)
#print(user_recommendations)
# Save user_recommendations to a pickle file
with open('LLM_recommendations.pkl', 'wb') as f:
    pickle.dump(responses, f)


#user_recommendations, test_items_dict = map_titles_to_ids(items_df, user_recommendations, df_items)
