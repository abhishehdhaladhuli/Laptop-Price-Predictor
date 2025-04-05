from flask import Flask, request, jsonify, render_template
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

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get form data as dictionary
            form_data = request.form.to_dict()

            # Convert numeric fields to float
            for key in ['inches', 'ram', 'weight']:
                form_data[key] = float(form_data[key])

            # Create DataFrame
            df = pd.DataFrame([form_data])

            # Apply label encoders
            for col in df.columns:
                if col in label_encoders:
                    df[col] = label_encoders[col].transform(df[col])

            prediction = model.predict(df)[0]
            return render_template("index.html", prediction=round(prediction, 2))

        except Exception as e:
            return render_template("index.html", error=str(e))

    # GET request - render the form
    return render_template("index.html")
