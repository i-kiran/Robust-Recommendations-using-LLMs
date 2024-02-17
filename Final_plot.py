from eval_LLM import sorted_auc_scores,combined_ranks
from data_preprocess import ratings_df 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_sparsity_scores(combined_ranks,ratings_df):
    pivot_table = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')
    sparsity_values = {}
    user_ids = list(combined_ranks.keys())
    for user_id in user_ids:
        user_ratings = pivot_table.loc[user_id]
        sparsity = user_ratings.isnull().sum() / len(user_ratings)
        sparsity_values[user_id] = sparsity

    sorted_sparsity_scores = dict(sorted(sparsity_values.items(), key=lambda item: item[1], reverse=True))
    return sorted_sparsity_scores

def plot_auc_vs_sparsity(sorted_auc_scores, sorted_sparsity_scores):
    # Extract the user_ids, auc_scores and sparsity_scores
    user_ids = sorted_auc_scores.keys()
    auc_scores = sorted_auc_scores.values()
    sparsity_scores = sorted_sparsity_scores.values()
    df = pd.DataFrame({
        'user_id': list(sorted_sparsity_scores.keys()),
        'sparsity': list(sorted_sparsity_scores.values()),
        'auc_score': [sorted_auc_scores[user_id] for user_id in sorted_sparsity_scores.keys()]
    })

    plt.figure(figsize=(10, 6), dpi=150)
    sns.set_style("whitegrid")
    df['distance_from_y_axis'] = abs(df['sparsity'])
    scatter = sns.scatterplot(data=df, x='sparsity', y='auc_score', hue='auc_score', palette='viridis', marker='.', size='distance_from_y_axis', sizes=(450, 70), edgecolor='black', linewidth=0.09)
    scatter.legend_.remove()

    plt.title('Sparsity values vs AUC scores for each user', fontsize=16)
    plt.xlabel('Sparsity value', fontsize=14)
    plt.ylabel('AUC score', fontsize=14)

    average_sparsity = df['sparsity'].mean()
    filtered_df = df[(df['auc_score'] <= 0.5) & (df['sparsity'] > average_sparsity)]

    plt.show()
    return filtered_df



sorted_sparsity_scores = calculate_sparsity_scores(combined_ranks,ratings_df)
print(sorted_auc_scores)
print(sorted_sparsity_scores)
# Call the function with your data
plot_auc_vs_sparsity(sorted_auc_scores, sorted_sparsity_scores)