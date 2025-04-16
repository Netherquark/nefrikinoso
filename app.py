import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

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

all_features = [c for c in _df.columns if c != 'class' and c != 'stage']
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'grf', 'sex', 'hypertension']
cat_cols = [c for c in cat_cols if c in all_features]
num_cols = [c for c in all_features if c not in cat_cols]

# Label Encoding
default_labels = {}
encoders = {}
for col in cat_cols:
    _df[col] = _df[col].astype(str).str.strip().str.lower()
    default = _df[col].mode()[0]
    default_labels[col] = default
    le = LabelEncoder()
    le.fit(_df[col])
    encoders[col] = le

# Prepare encoded version for model evaluation
_df_enc = _df.copy()
for col, le in encoders.items():
    _df_enc[col] = le.transform(_df_enc[col].str.strip().str.lower())
scaler = StandardScaler()
scaler.fit(_df_enc[all_features])

# Import Models
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
from voting_ensemble import VotingEnsembleModel

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
    "Voting": VotingEnsembleModel
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

# Evaluate each model by ROC AUC
def evaluate_model_roc_auc(model):
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(_df_enc[all_features])[:, 1]
            return roc_auc_score(_df_enc['class'], y_proba)
        return 0
    except Exception:
        return 0

# Select the best model by ROC AUC
best_model_name = max(models, key=lambda name: evaluate_model_roc_auc(models[name]))
best_model = models[best_model_name]


@app.route('/', methods=['GET', 'POST'])
def index():
    final_conclusion = None

    if request.method == 'POST':
        user_data = {feat: request.form.get(feat, '').strip().lower() for feat in all_features}
        df_input = pd.DataFrame([user_data], columns=all_features)

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
        pred = best_model.predict(X_input)[0]
        final_conclusion = "CKD" if pred == 0 else "NOT_CKD"

    return render_template('index.html', feature_order=all_features, final_conclusion=final_conclusion)


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
    pred = best_model.predict(X_input)[0]
    result = "CKD" if pred == 0 else "NOT_CKD"
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(debug=True)
