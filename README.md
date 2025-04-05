# CodeTech-DataScience-Task-3 Implementation of complete data science project from data preprocessing to deployment using flask
COMPANY: CODETECH IT SOLUTIONS <br>
NAME: ABHISHEK DHALADHULI<br>
INTERN ID: CT6WVEQ<br>
DOMAIN: DATA SCIENCE<br>
DURATION: 6 WEEKS<br>
MENTOR: NEELA SANTHOSH<br>
# DESCRIPTION OF THE TASK-3
Hi Abhi! Let's take a deep breath — I’ve got you covered now with a **full structured explanation** starting with a **proper project introduction**, followed by **step-by-step code explanation** from **data preprocessing to model training to Flask deployment**. 💻🚀

---

## 💡 **Project Introduction: Laptop Price Predictor**

This project is an end-to-end **machine learning web application** that predicts the **price of a laptop** based on user-input features like brand, CPU, RAM, screen resolution, and more. It includes the following stages:

1. **Data Preprocessing:** Cleaning and transforming raw data.
2. **Model Training:** Training a regression model using `RandomForestRegressor`.
3. **Model Saving:** Exporting the model and label encoders using `joblib`.
4. **Web App with Flask:** Creating a form-based web app where users enter laptop specs and get a predicted price.
5. **Deployment:** Hosting the app using **Render.com** for public access.

---

## 🔧 Part 1: Data Preprocessing & Model Training (`train_model.py`)

### 📌 Step 1: Import Necessary Libraries
```python
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
```
These libraries help with:
- **Data handling**: `pandas`, `numpy`
- **ML tools**: `train_test_split`, `RandomForestRegressor`, `LabelEncoder`, `metrics`
- **Saving models**: `joblib`
- **File system management**: `os`

---

### 📌 Step 2: Define Training Function
```python
def preprocess_and_train(filepath, target_col='price'):
```
- This function handles everything: cleaning data, training the model, and saving it.
- `filepath`: path to the CSV file.
- `target_col`: target variable (`price` by default).

---

### 📌 Step 3: Load and Clean the Data
```python
df = pd.read_csv(filepath)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.dropna(subset=[target_col])
```
- Loads CSV into a DataFrame.
- Removes unnamed (junk) columns.
- Standardizes column names.
- Removes rows where price is missing.

---

### 📌 Step 4: Process RAM, Weight, and Inches
```python
df['ram'] = df['ram'].astype(str).str.replace('GB', '').astype(float)
df['weight'] = df['weight'].astype(str).str.replace('kg', '').replace('?', np.nan)
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df['inches'] = pd.to_numeric(df['inches'], errors='coerce')
df = df.dropna()
```
- Removes unit suffixes like `GB` and `kg`.
- Converts to proper float types.
- Handles missing or malformed data.

---

### 📌 Step 5: Label Encoding (Converting Strings to Numbers)
```python
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
```
- Converts categorical columns (like company, CPU, GPU) into numbers using `LabelEncoder`.
- Stores each encoder to reuse later (during prediction).

---

### 📌 Step 6: Train-Test Split
```python
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Splits the data into 80% training and 20% testing sets.

---

### 📌 Step 7: Train the Random Forest Model
```python
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
```
- Trains a Random Forest model on the training data.

---

### 📌 Step 8: Evaluate the Model
```python
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```
- Evaluates model performance using RMSE and R² Score.

---

### 📌 Step 9: Save the Model and Encoders
```python
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/trained_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
```
- Saves the model and encoders to the `model` directory.

---

## 🌐 Part 2: Flask App (Backend + Frontend)

### 📌 Flask Backend (`app.py`)
```python
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model/trained_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
```
- Initializes Flask app.
- Loads the trained model and encoders.

---

### 📌 Define Route & Prediction Logic
```python
@app.route("/", methods=["GET", "POST"])
def index():
```
- Handles both GET (show form) and POST (get prediction) requests.

---

### 📌 Receive Form Data & Make Prediction
```python
if request.method == "POST":
    try:
        input_data = {
            "company": request.form["company"],
            ...
        }
        df = pd.DataFrame([input_data])
```
- Collects user input.
- Converts it to a DataFrame for prediction.

---

### 📌 Encode User Input
```python
for col in df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        value = df[col].iloc[0]
        if value not in le.classes_:
            le.classes_ = np.append(le.classes_, value)
        df[col] = le.transform(df[col])
```
- Transforms string inputs using saved encoders.
- Handles unseen labels by dynamically adding them.

---

### 📌 Predict Price
```python
pred = model.predict(df)[0]
prediction = round(pred, 2)
```
- Makes the prediction and rounds it for display.

---

### 📌 Render Result
```python
return render_template("index.html", prediction=prediction, error=error)
```
- Displays the prediction or error in the UI.

---

## 🎨 Part 3: HTML Frontend (`templates/index.html`)
- A stylish HTML form using pure CSS animations.
- Input fields for all laptop specs.
- Shows predicted price after submission.

---

## 🚀 Part 4: Deployment on Render
- Zip your project folder (`model/`, `templates/`, `app.py`, etc.)
- Push it to GitHub.
- Go to [Render.com](https://render.com)
  - Choose "New Web Service"
  - Connect to your GitHub repo
  - Set **build command** as `pip install -r requirements.txt`
  - Set **start command** as `python app.py`
- Deploy and get a public URL for your app.

---
