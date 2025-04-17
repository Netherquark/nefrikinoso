import json
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from predict_ckd import CKDPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATASET_PATH       = "ckd_prediction_dataset.csv"
BEST_MODEL_PATH    = "best_model.pkl"
FEATURE_ORDER_PATH = "feature_order.json"

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)  # Set Flask's logger level

# Human-friendly labels
# Human-friendly labels
display_names = {
    "age": "Age\n[years: 1–100]",
    "bp": "Blood Pressure\n[mmHg: 70–90]",
    "sg": "Specific Gravity\n[1.005–1.030]",
    "al": "Albumin Level\n[0–5 scale]",
    "su": "Sugar in Urine\n[0–5 scale]",
    "rbc": "Red Blood Cells\n[normal/abnormal]",
    "pc": "Pus Cells\n[normal/abnormal]",
    "pcc": "Pus Cell Clumps\n[present/not present]",
    "ba": "Bacteria in Urine\n[present/not present]",
    "bgr": "Blood Glucose (Random)\n[mg/dL: 70–140]",
    "bu": "Blood Urea\n[mg/dL: 7–20]",
    "sc": "Serum Creatinine\n[mg/dL: 0.6–1.3]",
    "sod": "Sodium\n[mmol/L: 135–145]",
    "pot": "Potassium\n[mmol/L: 3.5–5.0]",
    "hemo": "Hemoglobin\n[g/dL: F 12–16 | M 14–18]",
    "pcv": "Packed Cell Volume\n[%: 23–48]",
    "wbcc": "White Blood Cell Count\n[/µL: 4500–11000]",
    "rbcc": "Red Blood Cell Count\n[million/µL: 4.2–5.9]",
    "htn": "Hypertension\n[yes/no]",
    "dm": "Diabetes Mellitus\n[yes/no]",
    "cad": "Coronary Artery Disease\n[yes/no]",
    "appet": "Appetite\n[good/poor]",
    "pe": "Pedal Edema\n[yes/no]",
    "ane": "Anemia\n[yes/no]",
    "grf": "Glomerular Filtration Rate\n(estimated)",
    "hypertension": "Hypertension\n[yes/no]" # Added as it's in XGBoost's cat_cols
}

# 1) Load feature order
feature_order = []
try:
    with open(FEATURE_ORDER_PATH, "r") as f:
        feature_order = json.load(f)
    app.logger.info(f"Feature order loaded: {feature_order}")
except Exception as e:
    app.logger.error(f"Error loading feature_order.json: {e}")

# Define cat_cols here - should match the columns used during training
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm']

# 2) Load model & set up predictor
predictor = None
default_labels = {}
num_cols = []

if feature_order and cat_cols:
    try:
        with open(BEST_MODEL_PATH, "rb") as f:
            best_model = pickle.load(f)
        predictor = CKDPredictor(best_model, feature_order)
        predictor.setup_encoders_and_scaler()
        app.logger.info("CKDPredictor initialized.")
        app.logger.debug(f"Predictor encoders: {predictor.encoders}")
        app.logger.debug(f"Predictor scaler: {predictor.scaler}")

        # 3) Compute default_labels (mode) for each categorical AFTER predictor is initialized
        full = pd.read_csv(DATASET_PATH)
        for c in cat_cols:
            if c in predictor.encoders:
                vals = full[c].astype(str).str.lower().str.strip()
                default_labels[c] = vals.mode()[0] if not vals.mode().empty else predictor.encoders[c].classes_[0]
            else:
                app.logger.warning(f"Encoder not found for categorical column '{c}' during default label computation.")
        num_cols = [c for c in feature_order if c not in cat_cols]
        app.logger.debug(f"Categorical columns: {cat_cols}")
        app.logger.debug(f"Numerical columns: {num_cols}")
        app.logger.debug(f"Default labels: {default_labels}")

    except Exception as e:
        app.logger.error(f"Failed to init CKDPredictor: {e}")
else:
    app.logger.error("Feature order or categorical columns not loaded properly.")

@app.route("/", methods=["GET", "POST"])
def index():
    final_conclusion = None

    if request.method == "POST" and predictor:
        # 1) Collect raw inputs
        raw = {feat: request.form.get(feat, "").strip() for feat in feature_order}
        app.logger.debug(f"Raw input from form: {raw}")
        df  = pd.DataFrame([raw], columns=feature_order)
        app.logger.debug(f"DataFrame after raw input: \n{df}")

        # 2) Encode categoricals
        for c in cat_cols:
            if c in predictor.encoders: # Ensure the encoder exists
                le = predictor.encoders[c]
                v  = df.at[0, c].lower().strip()
                app.logger.debug(f"Encoding categorical column '{c}', value: '{v}'")
                # Handle potential NaN values in input
                if pd.isna(v) or v == 'nan':
                    v = 'missing'  # Use 'missing' as the placeholder for NaN
                if v not in le.classes_:
                    app.logger.warning(f"Value '{v}' not in classes for column '{c}'. Using default: '{default_labels.get(c, le.classes_[0] if le.classes_.size > 0 else 'missing')}'")
                    v = default_labels.get(c, le.classes_[0] if le.classes_.size > 0 else 'missing')
                df.at[0, c] = le.transform([v])[0]
                app.logger.debug(f"Encoded value for '{c}': {df.at[0, c]}")
            else:
                app.logger.warning(f"Encoder not found for categorical column: {c}")
        # ** REMOVE THIS LINE **
        # df[cat_cols] = df[cat_cols].astype(int)
        app.logger.debug(f"DataFrame after categorical encoding: \n{df}")

        # 3) Fill and scale numerical features
        numerical_cols = [col for col in feature_order if col not in cat_cols]
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median()) # Fill NaN in numerical cols

        try:
            df_scaled = predictor.scaler.transform(df[numerical_cols])
            df_scaled_df = pd.DataFrame(df_scaled, columns=numerical_cols, index=df.index)
            df = df.drop(columns=numerical_cols).merge(df_scaled_df, left_index=True, right_index=True)
            app.logger.debug(f"DataFrame after scaling: \n{df}")
        except Exception as e:
            app.logger.error(f"Error during scaling: {e}")

        # 4) (Optional) Log dtypes to verify
        app.logger.debug("Dtypes before predict:\n%s", df.dtypes.to_dict())

        # 5) Predict probability & final label
        try:
            proba = predictor.model.predict_proba(df[feature_order])[0, 1]
            app.logger.debug(f"Predicted probabilities: {predictor.model.predict_proba(df[feature_order])}")
            app.logger.debug(f"Probability of CKD: {proba}")
            final_conclusion = "CKD" if proba >= 0.5 else "No CKD"
            app.logger.info(f"Final conclusion: {final_conclusion}")
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            final_conclusion = "Error during prediction"

    return render_template(
        "index.html",
        feature_order=feature_order,
        display_names=display_names,
        final_conclusion=final_conclusion
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not predictor:
        return jsonify(error="Predictor not loaded"), 500

    data = request.get_json(force=True)
    app.logger.debug(f"API request data: {data}")
    df   = pd.DataFrame([data], columns=feature_order)

    # same encode/scale/cast
    for c in cat_cols:
        if c in predictor.encoders:
            le = predictor.encoders[c]
            v  = df.at[0, c].lower().strip()
            if pd.isna(v) or v == 'nan':
                v = 'missing'
            if v not in le.classes_:
                v = default_labels.get(c, le.classes_[0] if le.classes_.size > 0 else 'missing')
            df.at[0, c] = le.transform([v])[0]
    # ** REMOVE THIS LINE **
    # df[cat_cols] = df[cat_cols].astype(int)

    numerical_cols = [col for col in feature_order if col not in cat_cols]
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    df_scaled = predictor.scaler.transform(df[numerical_cols])
    df_scaled_df = pd.DataFrame(df_scaled, columns=numerical_cols, index=df.index)
    df = df.drop(columns=numerical_cols).merge(df_scaled_df, left_index=True, right_index=True)

    proba = predictor.model.predict_proba(df[feature_order])[0, 1]
    label = "Chronic Kidney Disease" if proba >= 0.5 else "No Chronic Kidney Disease"
    app.logger.info(f"API prediction result: {label}")
    return jsonify(result=label)

if __name__ == "__main__":
    app.run(debug=True)