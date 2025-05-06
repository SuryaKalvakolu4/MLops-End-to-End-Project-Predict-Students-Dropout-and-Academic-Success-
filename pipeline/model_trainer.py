# pipeline/model_trainer.py

import os
import joblib
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Model imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class ModelTrainer:
    def __init__(self, output_dir: str = "models"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models_with_params = self._define_models()
        self.results = []
        self.best_model = None
        self.best_model_name = ""
        self.best_f1_score = 0

    def _define_models(self) -> Dict[str, Tuple[Any, Dict]]:
        return {
            "RandomForest": (RandomForestClassifier(random_state=42), {
                "n_estimators": [100, 300, 500],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"]
            }),
            "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
                "n_estimators": [100, 200, 400],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"]
            }),
            "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), {
                "C": [0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"]
            }),
            "SVC": (SVC(probability=True, random_state=42), {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }),
            "DecisionTree": (DecisionTreeClassifier(random_state=42), {
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"]
            }),
            "AdaBoost": (AdaBoostClassifier(random_state=42), {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.5, 1.0, 1.5]
            }),
            "KNN": (KNeighborsClassifier(), {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }),
            "GaussianNB": (GaussianNB(), {
                "var_smoothing": [1e-09, 1e-08, 1e-07]
            })
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        for name, (model, param_grid) in self.models_with_params.items():
            print(f"\n🔍 Tuning {name}...")
            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=10,
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")

            model_path = f"{self.output_dir}/{name}_tuned_model.joblib"
            joblib.dump(best_model, model_path)

            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_model = best_model
                self.best_model_name = name

            self.results.append({
                "Model": name,
                "Accuracy": acc,
                "F1 Score": f1,
                "Precision": prec,
                "Recall": rec,
                "Best Params": search.best_params_
            })

        return pd.DataFrame(self.results)

    def get_best_model(self):
        return self.best_model_name, self.best_model

if __name__ == "__main__":
    print("ModelTrainer module is ready.")
