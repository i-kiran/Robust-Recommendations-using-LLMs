import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from metric_eval_RS import calculate_metrics
from train_model import ratings_df,trainset, testset, predictions

# Create a DataFrame from the sparsity_values and auc_scores dictionaries

def plot_sparsity_vs_auc(sorted_sparsity_values, sorted_auc_scores):
    df = pd.DataFrame({
        'user_id': list(sorted_sparsity_values.keys()),
        'sparsity': list(sorted_sparsity_values.values()),
        'auc_score': [sorted_auc_scores[user_id] for user_id in sorted_sparsity_values.keys()]
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

sorted_auc_scores, sorted_sparsity_scores = calculate_metrics(testset, predictions, ratings_df)
filtered_df = plot_sparsity_vs_auc(sorted_sparsity_scores, sorted_auc_scores)