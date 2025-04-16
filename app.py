import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Constants
dataset_path = "ckd_prediction_dataset.csv"
model_dir = "models"

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Initialize Flask App
app = Flask(__name__)

# Load full dataset to fit encoders and scaler
_df = pd.read_csv(dataset_path)
_df.drop(columns=[col for col in ['affected', 'age_avg', 'stage'] if col in _df.columns],
         inplace=True, errors='ignore')

# Identify features: exclude the class and any stage column
all_features = [c for c in _df.columns if c != 'class' and c != 'stage']
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'grf', 'sex',
            'hypertension']
cat_cols = [c for c in cat_cols if c in all_features]
num_cols = [c for c in all_features if c not in cat_cols]

# Fit label encoders on categorical columns and record default labels
default_labels = {}
encoders = {}
for col in cat_cols:
    _df[col] = _df[col].astype(str).str.strip().str.lower()
    default = _df[col].mode()[0]
    default_labels[col] = default
    le = LabelEncoder()
    le.fit(_df[col])
    encoders[col] = le

# Fit scaler on numeric+encoded categorical data
_df_enc = _df.copy()
for col, le in encoders.items():
    _df_enc[col] = le.transform(_df_enc[col].str.strip().str.lower())
scaler = StandardScaler()
scaler.fit(_df_enc[all_features])

# Load or train models
from XGBoost import XGBoostModel
from SVM import SVMModel
from Decision_tree import DecisionTreeModel
from LogisticRegression import LogisticRegressionModel
from knn import KNNModel
from NaiveBayes import NaiveBayesModel
from randomForest_classifier import RandomForestModel
from gradient_boost import GradientBoostModel
from catboost_ckd import CatBoostCKDModel
from stacked_ensemble import StackedEnsembleModel
from voting_ensemble import CKDEnsembleModel

model_classes = {
    "XGBoost": XGBoostModel,
    "SVM": SVMModel,
    "Decision Tree": DecisionTreeModel,
    "Logistic Regression": LogisticRegressionModel,
    "K-Nearest Neighbors": KNNModel,
    "Naive Bayes": NaiveBayesModel,
    "Random Forest": RandomForestModel,
    "Gradient Boosting": GradientBoostModel,
    "CatBoost": CatBoostCKDModel,
    "Stacked Ensemble Learning": StackedEnsembleModel,
    "Voting": CKDEnsembleModel
}

models = {}
for name, cls in model_classes.items():
    path = os.path.join(model_dir, f"{name.replace(' ', '_')}.joblib")
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        m = cls(dataset_path)
        m.preprocess_data()
        m.train_test_split()
        m.train_model()
        joblib.dump(m.model, path)
        models[name] = m.model


@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    final_conclusion = None

    if request.method == 'POST':
        # Collect user input from form fields
        user_data = {}
        for feat in all_features:
            user_data[feat] = request.form.get(feat, '').strip().lower()

        # Build DataFrame for input
        df_input = pd.DataFrame([user_data], columns=all_features)
        # Encode categorical features and convert to numerical
        for col, le in encoders.items():
            vals = df_input[col].astype(str).str.strip().str.lower()
            df_input[col] = [v if v in le.classes_ else default_labels[col] for v in vals]
            df_input[col] = le.transform(df_input[col])
        for col in num_cols:
            try:
                df_input[col] = df_input[col].astype(float)
            except ValueError:
                df_input[col] = 0.0
        X_input = scaler.transform(df_input[all_features])

        # Obtain prediction from Stacked Ensemble Model
        try:
            stacked_ensemble_model = models['Stacked Ensemble Learning']
            pred = stacked_ensemble_model.predict(X_input)[0]
            predictions['Stacked Ensemble Learning'] = "CKD" if pred == 0 else "NOT_CKD"
            final_conclusion = predictions['Stacked Ensemble Learning']

        except Exception as e:
            predictions['Stacked Ensemble Learning'] = f"Error: {e}"
            final_conclusion = "Error"

        return render_template('index.html', predictions=predictions,
                               feature_order=all_features,
                               final_conclusion=final_conclusion)

    # GET request: pass empty dictionaries
    return render_template('index.html', predictions=None, feature_order=all_features,
                           final_conclusion=None)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = {k: str(v).strip().lower() for k, v in request.json.items()}
    df_input = pd.DataFrame([data], columns=all_features)
    for col, le in encoders.items():
        vals = df_input[col].astype(str).str.strip().str.lower()
        df_input[col] = [v if v in le.classes_ else default_labels[col] for v in vals]
        df_input[col] = le.transform(df_input[col])
    for col in num_cols:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)
    X_input = scaler.transform(df_input[all_features])
    # Predict using Stacked Ensemble model
    result = {"Stacked Ensemble Learning": (
        "CKD" if models['Stacked Ensemble Learning'].predict(X_input)[0] == 0 else "NOT_CKD")}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)