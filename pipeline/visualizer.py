# pipeline/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_comparison(results_df: pd.DataFrame, save_path="outputs/model_comparison.png"):
    """
    Plot bar charts comparing models on F1 score, accuracy, etc.
    :param results_df: DataFrame with model evaluation metrics
    :param save_path: Path to save the comparison plot
    """
    metrics = ["F1 Score", "Accuracy", "Precision", "Recall"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Model", y=metric, data=results_df.sort_values(by=metric, ascending=False))
        plt.title(f"Model Comparison - {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path.replace(".png", f"_{metric.replace(' ', '_').lower()}.png"))
        plt.close()
