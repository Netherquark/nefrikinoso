import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import pickle  # For loading the pickled model

# Constants
dataset_path = "ckd_prediction_dataset.csv"
model_dir = "models"
BEST_MODEL_PATH = "best_model.pkl"
FEATURE_ORDER_PATH = "feature_order.json"

# Ensure model directory exists (though we might not use it directly anymore)
os.makedirs(model_dir, exist_ok=True)

# Initialize Flask App
app = Flask(__name__)

# Define the display mapping for GUI labels.
# Update these to match the new dataset's column names.
display_names = {
    "age": "Age\n[years: 1–100]",
    "bp": "Blood Pressure\n[mmHg: 70-90]",
    "sg": "Specific Gravity\n[1.005–1.030]",
    "al": "Albumin Level\n[0–5 scale]",
    "su": "Sugar in Urine\n[0–5 scale]",
    "rbc": "Red Blood Cells\n[normal/abnormal]",
    "pc": "Pus Cells\n[normal/abnormal]",
    "pcc": "Pus Cell Clumps\n[present/notpresent]",
    "ba": "Bacteria in Urine\n[present/notpresent]",
    "bgr": "Blood Glucose (Random)\n[mg/dL: 70–140]",
    "bu": "Blood Urea\n[mg/dL: 7–20]",
    "sc": "Serum Creatinine\n[mg/dL: 0.6–1.3]",
    "sod": "Sodium\n[mmol/L: 135–145]",
    "pot": "Potassium\n[mmol/L: 3.5–5.0]",
    "hemo": "Hemoglobin\n[g/dL: F 12–16 | M 14–18]",
    "pcv": "Packed Cell Volume\n[%: 23–48]",
    "wbcc": "White Blood Cell Count\n[/µL: 4500–11000]",
    "rbcc": "Red Blood Cell Count\n[million/µL: 4.2–5.9]",
    "htn": "Hypertension\n[yes/no]",
    "dm": "Diabetes Mellitus\n[yes/no]",
    "class": "Class" # Keep for internal use if needed
}

# Load feature order from JSON (load it at the top level)
try:
    with open(FEATURE_ORDER_PATH, 'r') as f:
        all_features = json.load(f)
except FileNotFoundError:
    print(f"Error: {FEATURE_ORDER_PATH} not found. Please run main.py first.")
    all_features = []
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {FEATURE_ORDER_PATH}.")
    all_features = []

# Load full dataset to fit encoders and scaler based on the new dataset
try:
    _df = pd.read_csv(dataset_path)
    # Drop the index column if it exists and is not needed
    if 'index' in _df.columns:
        _df.drop(columns=['index'], inplace=True)
except FileNotFoundError:
    print(f"Error: {dataset_path} not found.")
    _df = pd.DataFrame(columns=all_features + ['class']) # Create an empty DataFrame

# Define categorical and numerical features based on the new dataset
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm']
cat_cols = [c for c in cat_cols if c in all_features]
num_cols = [c for c in all_features if c not in cat_cols]

# Label Encoding for categorical variables
default_labels = {}
encoders = {}
for col in cat_cols:
    if col in _df.columns:
        _df[col] = _df[col].astype(str).str.strip().str.lower()
        default = _df[col].mode()[0] if not _df[col].empty else ''
        default_labels[col] = default
        le = LabelEncoder()
        le.fit(_df[col])
        encoders[col] = le

# Prepare encoded version for scaler fitting
_df_enc = _df.copy()
for col, le in encoders.items():
    if col in _df_enc.columns:
        _df_enc[col] = _df_enc[col].astype(str).str.strip().str.lower()
        # Handle cases where a label might not be in the training data
        _df_enc[col] = _df_enc[col].apply(lambda x: x if x in le.classes_ else default_labels.get(col, ''))
        _df_enc[col] = le.transform(_df_enc[col])

scaler = StandardScaler()
# Fit scaler only on numerical columns present in the loaded dataset
numerical_cols_present = [col for col in num_cols if col in _df_enc.columns]
if numerical_cols_present:
    scaler.fit(_df_enc[numerical_cols_present])

# Load the best model
best_model = None
try:
    with open(BEST_MODEL_PATH, 'rb') as f:
        best_model = pickle.load(f)
    print(f"Loaded best model from {BEST_MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: {BEST_MODEL_PATH} not found. Please run main.py first.")
except Exception as e:
    print(f"Error loading the best model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    final_conclusion = None
    predictions = {}

    if request.method == 'POST' and best_model is not None and all_features:
        user_data = {feat: request.form.get(feat, '').strip().lower() for feat in all_features}
        df_input = pd.DataFrame([user_data], columns=all_features)

        # Process categorical features
        for col, le in encoders.items():
            if col in df_input.columns:
                vals = df_input[col].astype(str).str.strip().str.lower()
                df_input[col] = [v if v in le.classes_ else default_labels.get(col, '') for v in vals]
                df_input[col] = le.transform(df_input[col])

        # Process numerical features
        for col in num_cols:
            if col in df_input.columns:
                try:
                    df_input[col] = df_input[col].astype(float)
                except ValueError:
                    df_input[col] = 0.0

        # Ensure all expected features are present (handle potential missing columns)
        for col in all_features:
            if col not in df_input.columns:
                df_input[col] = 0  # Or some appropriate default value

        # Scale numerical features
        numerical_cols_to_scale = [col for col in numerical_cols_present if col in df_input.columns]
        if numerical_cols_to_scale:
            df_input[numerical_cols_to_scale] = scaler.transform(df_input[numerical_cols_to_scale])

        # Make prediction
        try:
            pred = best_model.predict(df_input[all_features])[0]
            final_conclusion = "Chronic Kidney Disease" if pred == 1 else "No Chronic Kidney Disease"
        except Exception as e:
            final_conclusion = f"Error during prediction: {e}"

    return render_template(
        'index.html',
        feature_order=all_features,
        final_conclusion=final_conclusion,
        display_names=display_names,
        predictions=predictions
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if best_model is None or not all_features:
        return jsonify({"error": "Model or feature order not loaded."}), 500

    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided."}), 400

    df_input = pd.DataFrame([data], columns=all_features)

    # Process categorical features
    for col, le in encoders.items():
        if col in df_input.columns:
            vals = df_input[col].astype(str).str.strip().str.lower()
            df_input[col] = [v if v in le.classes_ else default_labels.get(col, '') for v in vals]
            df_input[col] = le.transform(df_input[col])

    # Process numerical features
    for col in num_cols:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)

    # Ensure all expected features are present
    for col in all_features:
        if col not in df_input.columns:
            df_input[col] = 0  # Or some appropriate default value

    # Scale numerical features
    numerical_cols_to_scale = [col for col in numerical_cols_present if col in df_input.columns]
    if numerical_cols_to_scale:
        df_input[numerical_cols_to_scale] = scaler.transform(df_input[numerical_cols_to_scale])

    try:
        pred = best_model.predict(df_input[all_features])[0]
        result = "Chronic Kidney Disease" if pred == 1 else "No Chronic Kidney Disease"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)