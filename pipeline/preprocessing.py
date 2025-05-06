# pipeline/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List

class Preprocessor:
    def __init__(self, numerical_cols: List[str], categorical_cols: List[str]):
        """
        Initialize the preprocessor with column types.
        :param numerical_cols: List of numerical column names
        :param categorical_cols: List of categorical column names
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.transformer = None

    def build_pipeline(self):
        """
        Create the preprocessing pipeline using ColumnTransformer.
        :return: ColumnTransformer
        """
        self.transformer = ColumnTransformer([
            ('num', StandardScaler(), self.numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
        ])
        return self.transformer

    def fit_transform(self, X: pd.DataFrame):
        """
        Fit and transform the input data.
        :param X: Input DataFrame
        :return: Transformed features
        """
        if not self.transformer:
            self.build_pipeline()
        return self.transformer.fit_transform(X)

    def transform(self, X: pd.DataFrame):
        """
        Transform data using the already fitted transformer.
        :param X: Input DataFrame
        :return: Transformed features
        """
        if not self.transformer:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return self.transformer.transform(X)

    def get_feature_names(self):
        """
        Get output feature names after transformation.
        :return: List of feature names
        """
        if self.transformer and hasattr(self.transformer, 'get_feature_names_out'):
            return self.transformer.get_feature_names_out()
        return []

if __name__ == "__main__":
    import pandas as pd

    # Sample usage
    df = pd.read_csv("data/data.csv", delimiter=";")
    df.columns = df.columns.str.replace(r'[\t\n]', '', regex=True).str.strip()
    X = df.drop(columns=["Target"])

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numerical_cols]

    preprocessor = Preprocessor(numerical_cols, categorical_cols)
    X_transformed = preprocessor.fit_transform(X)
    print("Preprocessing completed. Transformed shape:", X_transformed.shape)
