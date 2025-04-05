from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/trained_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            form_data = request.form.to_dict()

            # Convert numeric fields
            form_data["inches"] = float(form_data["inches"])
            form_data["ram"] = int(form_data["ram"])
            form_data["weight"] = float(form_data["weight"])

            df = pd.DataFrame([form_data])

            # Apply label encoders
            for col in df.columns:
                if col in label_encoders:
                    df[col] = label_encoders[col].transform(df[col])

            prediction = model.predict(df)[0]
            return render_template("index.html", prediction=round(prediction, 2))

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
