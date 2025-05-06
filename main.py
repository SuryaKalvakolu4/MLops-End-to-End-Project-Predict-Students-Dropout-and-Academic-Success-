# main.py (Project entry point)

from pipeline.data_loader import DataLoader
from pipeline.preprocessing import Preprocessor
from pipeline.model_trainer import ModelTrainer
from pipeline.evaluator import ModelEvaluator
from pipeline.visualizer import plot_model_comparison
import joblib
import os

if __name__ == "__main__":
    print("🚀 Starting MLOps pipeline...")

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

    # Step 3: Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42)

    # Step 4: Train and evaluate models
    trainer = ModelTrainer()
    results_df = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    best_model_name, best_model = trainer.get_best_model()

    # Step 5: Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    results_df.to_csv("outputs/model_metrics.csv", index=False)

    # Step 6: Evaluation and Visualization
    evaluator = ModelEvaluator(best_model, label_encoder, preprocessor)
    evaluator.plot_confusion_matrix(X, y, save_path="outputs/confusion_matrix.png")
    evaluator.plot_feature_importance(save_path="outputs/feature_importance.png")
    plot_model_comparison(results_df)

    print(f"✅ Pipeline completed. Best model: {best_model_name}")
