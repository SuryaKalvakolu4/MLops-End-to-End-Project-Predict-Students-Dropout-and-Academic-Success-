import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.schema import PredictionRequest
from app.model_loader import load_model, load_preprocessor, load_label_encoder
import pandas as pd

print("🚀 FastAPI main.py loaded")
app = FastAPI(title="Student Success Predictor")

# ✅ Mount static folder
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>🎓 Student Success Predictor is Running!</h2>
    <p>Use the <code>/predict</code> endpoint to send POST requests.</p>
    """

@app.post("/predict")
def predict(data: PredictionRequest):
    import traceback
    print("✅ /predict endpoint called")

    try:
        model = load_model("models/best_model.joblib")
        preprocessor = load_preprocessor("models/preprocessor.joblib")
        label_encoder = load_label_encoder("models/label_encoder.joblib")

        df = pd.DataFrame([data.dict()])
        print("📦 DataFrame received:\n", df)

        # Let's inspect shape first
        transformed = preprocessor.transform(df)
        print("🔄 Transformed shape:", transformed.shape)

        # Check model input shape
        pred_encoded = model.predict(transformed)[0]
        print("📊 Encoded prediction:", pred_encoded)

        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        print("✅ Final prediction:", prediction)

        return {"prediction": prediction}

    except Exception as e:
        print("❌ Prediction failed:")
        traceback.print_exc()
        return {"prediction": "error", "detail": str(e)}
