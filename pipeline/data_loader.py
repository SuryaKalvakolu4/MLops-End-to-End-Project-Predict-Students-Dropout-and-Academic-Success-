# pipeline/data_loader.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, filepath: str, target_column: str = 'Target'):
        """
        Initialize DataLoader with dataset path and target column.
        :param filepath: Path to the CSV file.
        :param target_column: Name of the target column.
        """
        self.filepath = filepath
        self.target_column = target_column
        self.df = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """
        Loads and cleans the dataset.
        :return: Pandas DataFrame
        """
        self.df = pd.read_csv(self.filepath, delimiter=';')
        self.df.columns = self.df.columns.str.replace(r'[\t\n]', '', regex=True).str.strip()
        return self.df

    def get_features_and_target(self):
        """
        Splits the dataset into features and target.
        :return: Tuple (X, y)
        """
        if self.df is None:
            raise ValueError("data not loaded. Call load_data() first.")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return X, y

    def encode_target(self, y):
        """
        Encodes the target variable using LabelEncoder.
        :param y: Target variable series
        :return: Tuple of encoded target and fitted LabelEncoder
        """
        y_encoded = self.label_encoder.fit_transform(y)
        return y_encoded, self.label_encoder

if __name__ == "__main__":
    # Example usage
    loader = DataLoader(filepath="data/data.csv", target_column="Target")
    df = loader.load_data()
    X, y = loader.get_features_and_target()
    y_encoded, le = loader.encode_target(y)
    print("data loaded and target encoded successfully.")
