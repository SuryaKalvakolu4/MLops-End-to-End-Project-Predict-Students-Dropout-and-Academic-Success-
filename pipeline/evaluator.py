# pipeline/evaluator.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class ModelEvaluator:
    def __init__(self, model, label_encoder, preprocessor):
        self.model = model
        self.label_encoder = label_encoder
        self.preprocessor = preprocessor

    def plot_confusion_matrix(self, X, y, save_path="confusion_matrix.png"):
        """
        Generate and save confusion matrix heatmap.
        :param X: Original input DataFrame (not transformed)
        :param y: True target values (not encoded)
        :param save_path: File path to save the confusion matrix image
        """
        X_transformed = self.preprocessor.transform(X)
        y_encoded = self.label_encoder.transform(y)
        y_pred = self.model.predict(X_transformed)

        cm = confusion_matrix(y_encoded, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def plot_feature_importance(self, save_path="feature_importance.png", top_n=20):
        """
        Plot and save feature importance if supported by the model.
        :param save_path: File path to save the feature importance image
        :param top_n: Number of top features to display
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.preprocessor.get_feature_names()
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(top_n)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title("Top Feature Importances")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Feature importance plot saved to {save_path}")
        else:
            print("⚠️ Feature importance not supported for this model.")

if __name__ == "__main__":
    print("Evaluator module loaded. Ready to generate visualizations.")
