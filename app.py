import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# Constants
dataset_path = "ckd_prediction_dataset.csv"
model_dir = "models"

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Initialize Flask App
app = Flask(__name__)

# Load full dataset to fit encoders and scaler
_df = pd.read_csv(dataset_path)
_df.drop(columns=[c for c in ['affected','age_avg','stage'] if c in _df.columns],
         inplace=True, errors='ignore')

# Identify features
all_features = [c for c in _df.columns if c not in ('class','stage')]
cat_cols = [c for c in ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','grf','sex','hypertension'] if c in all_features]
num_cols = [c for c in all_features if c not in cat_cols]

# Fit label encoders on categoricals
default_labels = {}
encoders = {}
for col in cat_cols:
    _df[col] = _df[col].astype(str).str.strip().str.lower()
    default_labels[col] = _df[col].mode()[0]
    encoders[col] = LabelEncoder().fit(_df[col])

# Fit scaler on all features (numerical + encoded categorical)
_df_enc = _df.copy()
for col, le in encoders.items():
    _df_enc[col] = le.transform(_df_enc[col])
scaler = StandardScaler().fit(_df_enc[all_features])

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
    "Stacked Ensemble": StackedEnsembleModel,
    "Voting Ensemble": CKDEnsembleModel
}

models = {}
for name, cls in model_classes.items():
    path = os.path.join(model_dir, f"{name.replace(' ','_')}.joblib")
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        m = cls(dataset_path)
        m.preprocess_data()
        m.train_test_split()
        m.train_model()
        joblib.dump(m.model, path)
        models[name] = m.model

# Compute recall for each model on a held‑out 20% split
_df_eval = pd.read_csv(dataset_path)
_df_eval.drop(columns=[c for c in ['affected','age_avg','stage'] if c in _df_eval.columns],
              inplace=True, errors='ignore')
_df_eval['class'] = _df_eval['class'].map({'ckd':1,'notckd':0})
for col, le in encoders.items():
    vals = _df_eval[col].astype(str).str.strip().str.lower()
    _df_eval[col] = [v if v in le.classes_ else default_labels[col] for v in vals]
    _df_eval[col] = le.transform(_df_eval[col])
for col in num_cols:
    _df_eval[col] = pd.to_numeric(_df_eval[col], errors='coerce').fillna(0.0)

X_all = scaler.transform(_df_eval[all_features])
y_all = _df_eval['class'].values
_, X_test_eval, _, y_test_eval = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

recalls = {}
for name, model in models.items():
    try:
        y_pred = model.predict(X_test_eval)
        recalls[name] = recall_score(y_test_eval, y_pred, average='macro')
    except Exception:
        recalls[name] = None

@app.route('/', methods=['GET','POST'])
def index():
    predictions = {}
    final_conclusion = None

    if request.method == 'POST':
        # Collect form data
        user_data = {feat: request.form.get(feat,'').strip().lower() for feat in all_features}
        df_input = pd.DataFrame([user_data], columns=all_features)

        # Encode & scale
        for col, le in encoders.items():
            vals = df_input[col].astype(str).str.strip().str.lower()
            df_input[col] = [v if v in le.classes_ else default_labels[col] for v in vals]
            df_input[col] = le.transform(df_input[col])
        for col in num_cols:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)

        X_input = scaler.transform(df_input[all_features])

        # Predict
        for name, model in models.items():
            try:
                p = model.predict(X_input)[0]
                predictions[name] = "CKD" if p==0 else "NOT_CKD"
            except Exception:
                predictions[name] = "Error"

        # Determine final conclusion by model with highest recall
        valid = {m: r for m, r in recalls.items() if r is not None}
        if valid:
            best_model = max(valid, key=valid.get)
            final_conclusion = predictions.get(best_model)

    return render_template(
        'index.html',
        predictions=predictions or None,
        recalls=recalls,
        model_names=list(models.keys()),
        feature_order=all_features,
        final_conclusion=final_conclusion
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = {k:str(v).strip().lower() for k,v in request.json.items()}
    df_input = pd.DataFrame([data], columns=all_features)
    for col, le in encoders.items():
        vals = df_input[col].astype(str).str.strip().str.lower()
        df_input[col] = [v if v in le.classes_ else default_labels[col] for v in vals]
        df_input[col] = le.transform(df_input[col])
    for col in num_cols:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)
    X_input = scaler.transform(df_input[all_features])
    result = {name: ("CKD" if m.predict(X_input)[0]==0 else "NOT_CKD") for name,m in models.items()}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
