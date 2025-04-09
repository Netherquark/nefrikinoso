import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

class CKDEnsembleModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None
        self.feature_names = []

        self.preprocess_data()
        self.train_test_split()
        self.train_model()

    def preprocess_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(['affected', 'age_avg'], axis=1, errors='ignore')

        # Encode categorical columns
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])  # CKD=0, NOT_CKD=1
        df['grf'] = le.fit_transform(df['grf'])

        self.X = df.drop('class', axis=1)
        self.y = df['class']
        self.feature_names = self.X.columns.tolist()

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self):
        lr = LogisticRegression(max_iter=1000)
        svm = SVC(kernel='rbf', probability=True)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        self.model = VotingClassifier(
            estimators=[('lr', lr), ('svm', svm), ('rf', rf), ('xgb', xgb)],
            voting='soft'
        )

        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self, return_scores=False):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        roc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        cm = confusion_matrix(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred)

        print("\nEvaluation for Voting Ensemble")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(cr)

        if return_scores:
            return {
                "Accuracy": acc,
                "ROC_AUC": roc,
                "y_true": self.y_test,
                "y_pred": y_pred
            }

        return None
