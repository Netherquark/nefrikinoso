import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class LogisticRegressionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def preprocess_data(self):
        # Load data
        self.df = pd.read_csv(self.file_path)

        # Drop unused columns
        drop_cols = ['index','affected', 'age_avg', 'stage']
        self.df.drop(columns=[col for col in drop_cols if col in self.df.columns], inplace=True)

        # Label encode target and 'grf' if needed
        le = LabelEncoder()
        if self.df['class'].dtype == 'object':
            self.df['class'] = le.fit_transform(self.df['class'])
        if 'grf' in self.df.columns and self.df['grf'].dtype == 'object':
            self.df['grf'] = le.fit_transform(self.df['grf'])

        # Set features and target
        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

        # Categorical column handling
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
                            'appet', 'pe', 'ane', 'grf', 'sex', 'hypertension']
        categorical_cols = [col for col in categorical_cols if col in self.X.columns]

        for col in categorical_cols:
            self.X[col] = self.X[col].astype(str).str.lower().str.strip()
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])

        # Fill missing values
        self.X = self.X.fillna(0)

        # Scale features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'penalty': ['l2']
        }

        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

        print("\nBest Parameters Found for Logistic Regression:")
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

        print(f"\nModel Evaluation for Logistic Regression:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.4f}")