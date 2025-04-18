import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import individual model wrappers
from XGBoost import XGBoostModel
from randomForest_classifier import RandomForestModel
from gradient_boost import GradientBoostModel
from SVM import SVMModel
from catboost_ckd import CatBoostCKDModel
from LogisticRegression import LogisticRegressionModel
from Decision_tree import DecisionTreeModel
from knn import KNNModel
from NaiveBayes import NaiveBayesModel


class StackedEnsembleModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.meta_model = None
        self.model = None

    def preprocess_data(self):
        self.df = pd.read_csv(self.dataset_path)

        # Drop unused columns
        drop_cols = ['index','affected', 'age_avg', 'stage']
        self.df.drop(columns=[col for col in drop_cols if col in self.df.columns], inplace=True)

        # Label encode target and 'grf' if needed
        le = LabelEncoder()
        if self.df['class'].dtype == 'object':
            self.df['class'] = le.fit_transform(self.df['class'])
        if 'grf' in self.df.columns and self.df['grf'].dtype == 'object':
            self.df['grf'] = le.fit_transform(self.df['grf'])

        # Features and target
        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

        # Encode categorical features
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
                            'appet', 'pe', 'ane', 'grf', 'sex', 'hypertension']
        categorical_cols = [col for col in categorical_cols if col in self.X.columns]
        for col in categorical_cols:
            self.X[col] = self.X[col].astype(str).str.lower().str.strip()
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])

        # Fill missing values and scale
        self.X = self.X.fillna(0)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )

    def tune_model(self, model, param_grid, X, y):
        grid = GridSearchCV(model, param_grid, scoring='recall', cv=5, n_jobs=-1)
        grid.fit(X, y)
        return grid.best_estimator_

    def train_model(self):
        model_classes = [
            (XGBoostModel, "XGBoost"),
            (RandomForestModel, "Random Forest"),
            (GradientBoostModel, "Gradient Boost"),
            (SVMModel, "SVM"),
            (CatBoostCKDModel, "CatBoost"),
            (LogisticRegressionModel, "Logistic Regression"),
            (DecisionTreeModel, "Decision Tree"),
            (KNNModel, "KNN"),
            (NaiveBayesModel, "Naive Bayes")
        ]

        print("Evaluating all base models...")
        model_scores = []
        for model_class, name in model_classes:
            try:
                model = model_class(self.dataset_path)
                model.preprocess_data()
                model.train_test_split()
                model.train_model()
                scores = model.evaluate_model(return_scores=True)
                model_scores.append((name, model, scores["Accuracy"], scores["ROC_AUC"]))
            except Exception as e:
                print(f"Skipping {name} due to error: {e}")

        top_models = sorted(model_scores, key=lambda x: x[2], reverse=True)[:3]
        print("\nTop 3 models selected for meta-learning:")
        for name, _, acc, auc in top_models:
            print(f"{name}: Accuracy={acc:.4f}, ROC_AUC={auc:.4f}")

        tuned_estimators = []
        for name, model, _, _ in top_models:
            base_model = model.model

            if name == "Random Forest":
                param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
            elif name == "XGBoost":
                param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            elif name == "SVM":
                param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            else:
                param_grid = {}

            if param_grid:
                best_estimator = self.tune_model(base_model, param_grid, self.X_train, self.y_train)
            else:
                best_estimator = base_model

            tuned_estimators.append((name.lower().replace(" ", "_"), best_estimator))

        # Meta-model tuning
        meta_param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
        final_estimator = LogisticRegression(max_iter=1000)
        tuned_meta = self.tune_model(final_estimator, meta_param_grid, self.X_train, self.y_train)

        self.meta_model = StackingClassifier(
            estimators=tuned_estimators,
            final_estimator=tuned_meta,
            passthrough=True,
            cv=5,
            n_jobs=-1
        )

        self.meta_model.fit(self.X_train, self.y_train)
        self.model = self.meta_model

    def evaluate_model(self, return_scores=False):
        y_pred = self.meta_model.predict(self.X_test)
        y_proba = self.meta_model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        recall = recall_score(self.y_test, y_pred)

        if return_scores:
            return {
                "Accuracy": accuracy,
                "ROC_AUC": roc_auc,
                "Recall": recall,
                "y_true": self.y_test,
                "y_pred": y_pred
            }

        print("\nMeta-Model Evaluation")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.4f}")
