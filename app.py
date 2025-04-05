from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/trained_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            input_data = {
                "company": request.form["company"],
                "typename": request.form["typename"],
                "inches": float(request.form["inches"]),
                "screenresolution": request.form["screenresolution"],
                "cpu": request.form["cpu"],
                "ram": int(request.form["ram"]),
                "memory": request.form["memory"],
                "gpu": request.form["gpu"],
                "opsys": request.form["opsys"],
                "weight": float(request.form["weight"])
            }

            df = pd.DataFrame([input_data])

            for col in df.columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    value = df[col].iloc[0]
                    
                    # Handle unseen label
                    if value not in le.classes_:
                        le.classes_ = np.append(le.classes_, value)
                    
                    df[col] = le.transform(df[col])

            pred = model.predict(df)[0]
            prediction = round(pred, 2)
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
