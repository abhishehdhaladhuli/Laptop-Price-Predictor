import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def preprocess_and_train(filepath, target_col='price'):
    # Load dataset
    df = pd.read_csv(filepath)

    # Clean columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop missing values in target
    df = df.dropna(subset=[target_col])

    # Handle units in ram, weight, inches
    if 'ram' in df.columns:
        df['ram'] = df['ram'].astype(str).str.replace('GB', '').astype(float)
    
    if 'weight' in df.columns:
        df['weight'] = df['weight'].astype(str).str.replace('kg', '').replace('?', np.nan)
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

    if 'inches' in df.columns:
        df['inches'] = pd.to_numeric(df['inches'], errors='coerce')

    # Drop any rows with missing values
    df = df.dropna()

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model trained")
    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📈 R² Score: {r2:.4f}")

    # Save model and encoders
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/trained_model.pkl")
    joblib.dump(label_encoders, "model/label_encoders.pkl")
    print("💾 Model & encoders saved in /model folder")

# Run the training
if __name__ == "__main__":
    preprocess_and_train("laptopData (1).csv")
