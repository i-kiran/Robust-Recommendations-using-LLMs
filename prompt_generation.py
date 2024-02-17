
from decision_plot import filtered_df
from train_model import trainset, testset
import random
import pandas as pd

def convert_to_dataframe(trainset, testset):
    # Convert train set to DataFrame
    ratings = []
    for user_id, item_id, rating in trainset.all_ratings():
        raw_user_id = trainset.to_raw_uid(user_id)
        raw_item_id = trainset.to_raw_iid(item_id)
        ratings.append([raw_user_id, raw_item_id, rating])
    df_train = pd.DataFrame(ratings, columns=['user_id', 'item_id', 'rating'])

    # Convert test set to DataFrame
    df_test = pd.DataFrame(testset, columns=['user_id', 'item_id', 'rating'])

    return df_train, df_test


import pandas as pd

def prepare_dataset(filtered_df, df_train, df_test):
    # Initialize lists to store the data
    user_ids = []
    train_items = []
    test_items = []

    # Iterate over the users in the filtered DataFrame
    for user_id in filtered_df['user_id']:
        # Find the items and their ratings that the user rated in the train set
        train_items_user = df_train[df_train['user_id'] == user_id][['item_id', 'rating']].values.tolist()
        
        # Find the items and their ratings that the user rated in the test set
        test_items_user = df_test[df_test['user_id'] == user_id][['item_id', 'rating']].values.tolist()
        
        # Append the data to the lists
        user_ids.append(user_id)
        train_items.append(train_items_user)
        test_items.append(test_items_user)

    # Create a new DataFrame with the data
    df_items = pd.DataFrame({
        'user_id': user_ids,
        'train_items': train_items,
        'test_items': test_items
    })

    # Define a function to sort a list of tuples by the second element in descending order
    def sort_by_rating(items):
        return sorted(items, key=lambda x: x[1], reverse=True)

    # Sort the tuples in the train_items and test_items columns by rating
    df_items['train_items'] = df_items['train_items'].apply(sort_by_rating)
    df_items['test_items'] = df_items['test_items'].apply(sort_by_rating)

    # Define a function to extract item ids from a list of tuples
    def extract_item_ids(items):
        return [item_id for item_id, rating in items]

    # Extract the item ids from the tuples in the train_items and test_items columns
    df_items['train_items'] = df_items['train_items'].apply(extract_item_ids)
    df_items['test_items'] = df_items['test_items'].apply(extract_item_ids)

    return df_items

def generate_prompts(df_items):
    # Initialize a list to store the prompts
    prompts = []

    # Iterate over the rows in the DataFrame
    for _, row in df_items.iterrows():
        # Randomly select the top 40 items from the train_items and test_items columns
        train_items = random.sample(row['train_items'][:20], min(20, len(row['train_items'])))
        test_items = random.sample(row['test_items'][:180], min(180, len(row['test_items'])))
        
        # Format the prompt
        prompt = f"User {row['user_id']} rated following items in decreasing order of preference {train_items}. Rank following items in the decreasing preference such that item on top should be the most liked one {test_items}.Do not suggest any movie outside this list. Also, do not generate anything else in the messae but only the list and do not forget to use 'user_id' and 'recommnded_items' as keys in the message."
        
        # Append the prompt to the list
        prompts.append(prompt)
    
    return prompts

print(filtered_df.head())
df_train, df_test = convert_to_dataframe(trainset, testset)
df_items = prepare_dataset(filtered_df, df_train, df_test)
prompts = generate_prompts(df_items)