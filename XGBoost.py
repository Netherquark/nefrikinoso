import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap


class XGBoostModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.model = None

    def preprocess_data(self):
        # Remove redundant columns
        self.df = self.df.drop(['affected', 'age_avg'], axis=1)

        # Encode categorical columns
        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])
        self.df['grf'] = le.fit_transform(self.df['grf'])

        # Define features and target
        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

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

        print("\nBest Parameters:", grid_search.best_params_)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Accuracy and Classification Report
        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        classification = classification_report(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        print("\nAccuracy:", accuracy)
        print("\nConfusion Matrix:\n", confusion)
        print("\nClassification Report:\n", classification)
        print("\nROC AUC Score:", roc_auc)

        # Plot visualizations
        self.plot_feature_importance()
        self.plot_roc_curve(y_proba)
        self.plot_precision_recall_curve(y_proba)
        self.plot_learning_curve()
        self.plot_shap_values()
        self.plot_feature_distribution()

    def plot_feature_importance(self):
        importance = self.model.feature_importances_
        features = self.X.columns

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features, palette='viridis')
        plt.title('Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.show()

    def plot_roc_curve(self, y_proba):
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc_score(self.y_test, y_proba):.3f}')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.title('ROC Curve', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend()
        plt.show()

    def plot_precision_recall_curve(self, y_proba):
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', label='Precision-Recall Curve')
        plt.title('Precision-Recall Curve', fontsize=16)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.legend()
        plt.show()

    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=5,
            scoring='roc_auc', train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
        plt.plot(train_sizes, test_mean, label='Cross-Validation Score', color='green')
        plt.title('Learning Curve', fontsize=16)
        plt.xlabel('Training Size', fontsize=12)
        plt.ylabel('ROC AUC Score', fontsize=12)
        plt.legend()
        plt.show()

    def plot_shap_values(self):
        explainer = shap.Explainer(self.model, self.X_test)
        shap_values = explainer(self.X_test)

        plt.figure(figsize=(12, 6))
        shap.plots.beeswarm(shap_values)
        plt.title('SHAP Values', fontsize=16)
        plt.show()

    def plot_feature_distribution(self):
        plt.figure(figsize=(12, 8))
        for column in self.X.columns[:5]:  # Plot first 5 features for better visibility
            sns.kdeplot(self.df[column][self.df['class'] == 0], label=f"{column} - Class 0", shade=True)
            sns.kdeplot(self.df[column][self.df['class'] == 1], label=f"{column} - Class 1", shade=True)
            plt.title(f"Distribution of {column} by Class", fontsize=16)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    model = XGBoostModel('/Users/janhvidoijad/Desktop/Nefrikinoso/nefrikinoso/ckd_prediction_dataset.csv')
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
