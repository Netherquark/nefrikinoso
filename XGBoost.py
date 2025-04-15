import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class XGBoostModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.model = None

    def preprocess_data(self):
        columns_to_drop = [col for col in ['affected', 'age_avg', 'stage'] if col in self.df.columns]
        self.df = self.df.drop(columns=columns_to_drop, axis=1)

        le = LabelEncoder()
        if self.df['class'].dtype == 'object':
            self.df['class'] = le.fit_transform(self.df['class'])
        if self.df['grf'].dtype == 'object':
            self.df['grf'] = le.fit_transform(self.df['grf'])

        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

    def train_test_split(self, test_size=0.2, random_state=42):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 4],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }

        xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')

        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

        print("\n Best Parameters Found for XGBoost:")
        print(grid_search.best_params_)

    def evaluate_model(self, return_scores=False):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        if return_scores:
            return {
                "Accuracy": accuracy,
                "ROC_AUC": roc_auc
            }

        print(f"\n Model Evaluation for XGBoost:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.4f}")

        self.visualize_results(y_pred)

    def visualize_results(self, y_pred):
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Oranges')
        plt.title("Confusion Matrix - XGBoost")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
