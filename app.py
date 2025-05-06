from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("models/best_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # Convert types
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass  # skip string fields like Gender or Marital_status

        X = preprocessor.transform(df)
        pred_encoded = model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        return render_template("index.html", prediction=f"Predicted Status: {pred_label}")
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
