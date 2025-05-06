# app/model_loader.py

import joblib

def load_model(model_path: str):
    """Load a trained model from file."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def load_preprocessor(preprocessor_path: str):
    """Load a preprocessing transformer from file."""
    try:
        return joblib.load(preprocessor_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor: {e}")

def load_label_encoder(encoder_path: str):
    """Load a label encoder from file."""
    try:
        return joblib.load(encoder_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load label encoder: {e}")
