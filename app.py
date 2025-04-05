from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/trained_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

@app.route("/")
def home():
    return "Laptop Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame from input
        df = pd.DataFrame([data])

        # Preprocess input
        for col in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        prediction = model.predict(df)[0]
        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
