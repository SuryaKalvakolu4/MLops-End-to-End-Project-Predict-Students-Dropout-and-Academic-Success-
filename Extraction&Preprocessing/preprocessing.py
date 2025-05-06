# mlops_pipeline.py (Complete MLOps Pipeline + Visualization)

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline

# models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# =====================
# 1. Load and Clean data
# =====================
df = pd.read_csv(r"/data/data.csv", delimiter=";")
df.columns = df.columns.str.replace(r"[\t\n]", "", regex=True).str.strip()

X = df.drop(columns=["Target"])
y = df["Target"]

# =====================
# 2. Encode Target
# =====================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =====================
# 3. Identify Feature Types
# =====================
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numerical_cols]

# =====================
# 4. Preprocessing Pipeline
# =====================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])
X_preprocessed = preprocessor.fit_transform(X)

# =====================
# 5. Train-Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# =====================
# 6. Define models and Hyperparameters
# =====================
models_to_tune = {
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

# =====================
# 7. Train, Tune, Evaluate and Save models
# =====================
results = []
best_model = None
best_f1 = 0
best_model_name = ""

for name, (model, param_grid) in models_to_tune.items():
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
    tuned_model = search.best_estimator_

    # Evaluate
    y_pred = tuned_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    # Save model
    model_path = f"{name}_tuned_model.joblib"
    joblib.dump(tuned_model, model_path)

    # Track best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = tuned_model
        best_model_name = name

    # Log results
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": rec,
        "Best Params": search.best_params_
    })

# =====================
# 8. Display Results
# =====================
results_df = pd.DataFrame(results)
print("\n📊 Model Evaluation Summary:")
print(results_df.to_string(index=False))
results_df.to_csv("model_tuning_results.csv", index=False)

# =====================
# 9. Confusion Matrix and Feature Importance for Best Model
# =====================
import numpy as np

# Use full dataset for clearer confusion matrix
X_full = preprocessor.transform(X)
y_pred_full = best_model.predict(X_full)

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred_full)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{best_model_name}_confusion_matrix.png')
plt.show()

# Feature importance (if supported)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig(f'{best_model_name}_feature_importance.png')
    plt.show()
else:
    print(f"\nℹ️ Feature importance is not available for {best_model_name}.")
