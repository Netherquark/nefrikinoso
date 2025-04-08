import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class CatBoostCKDModel:
    def __init__(self, file_path, params=None):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        if params is None:
            params = {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.1,
                'loss_function': 'Logloss',
                'verbose': 100
            }
        self.model = CatBoostClassifier(**params)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self):
        self.df = self.df.drop(['affected', 'age_avg'], axis=1)

        # Encode categorical target
        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])

        # Convert numeric columns
        self.df['grf'] = pd.to_numeric(self.df['grf'], errors='coerce')

        X = self.df.drop('class', axis=1)
        y = self.df['class']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def hyperparameter_tuning(self):
        param_dist = {
            'iterations': [500, 1000, 1500],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
        random_search = RandomizedSearchCV(
            CatBoostClassifier(loss_function='Logloss', verbose=0),
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(self.X_train, self.y_train)
        print("Best parameters found:", random_search.best_params_)

        self.model = CatBoostClassifier(
            **random_search.best_params_,
            loss_function='Logloss',
            verbose=100
        )

    def train(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call preprocess_data() first.")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not loaded. Call preprocess_data() first.")
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        classification = classification_report(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        print("Accuracy:", accuracy)
        print("\nConfusion Matrix:\n", confusion)
        print("\nClassification Report:\n", classification)
        print("\nROC AUC Score:", roc_auc)

        self.visualize_results(y_pred, y_proba)

    def visualize_results(self, y_pred, y_proba):
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.plot(fpr, tpr, label="ROC Curve", color="darkorange")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

        # Feature Importance
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            features = self.X_train.columns
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
            plt.title("Feature Importance")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.show()
        elif hasattr(self.model, "coef_"):
            importances = self.model.coef_[0]
            features = self.X_train.columns
            indices = np.argsort(abs(importances))[::-1]
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="magma")
            plt.title("Feature Importance (Coefficients)")
            plt.xlabel("Coefficient Value")
            plt.ylabel("Features")
            plt.show()
        else:
            print("Feature importance not available for this model.")

if __name__ == "__main__":
    model = CatBoostCKDModel("ckd_prediction_dataset.csv")
    model.preprocess_data()
    model.hyperparameter_tuning()
    model.train()
    model.evaluate()
