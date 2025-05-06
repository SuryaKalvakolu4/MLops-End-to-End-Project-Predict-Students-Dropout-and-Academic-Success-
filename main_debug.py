# main_debug.py
# Train and serve model without saving to disk (for debugging only)

from pipeline.data_loader import DataLoader
from pipeline.preprocessing import Preprocessor
from pipeline.model_trainer import ModelTrainer
from pipeline.evaluator import ModelEvaluator
from pipeline.visualizer import plot_model_comparison
import numpy as np
import pandas as pd

print("⚙️ Running debug version of training pipeline...")

# Step 1: Load and prepare data
loader = DataLoader(filepath="data/data.csv")
df = loader.load_data()
X, y = loader.get_features_and_target()
y_encoded, label_encoder = loader.encode_target(y)

# Step 2: Preprocessing
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numerical_cols]
preprocessor = Preprocessor(numerical_cols, categorical_cols)
X_transformed = preprocessor.fit_transform(X)

# Step 3: Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42)

# Step 4: Train in-memory
trainer = ModelTrainer()
results_df = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
best_model_name, best_model = trainer.get_best_model()

# Step 5: Evaluate and visualize
print("✅ Best model selected:", best_model_name)
evaluator = ModelEvaluator(best_model, label_encoder, preprocessor)
evaluator.plot_confusion_matrix(X, y, save_path="outputs/confusion_matrix.png")
evaluator.plot_feature_importance(save_path="outputs/feature_importance.png")
plot_model_comparison(results_df)

print("🎯 Debug training completed.")
