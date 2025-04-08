import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class XGBoostModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.model = None

    def preprocess_data(self):
        # Drop unwanted columns
        self.df = self.df.drop(['affected', 'age_avg'], axis=1)

        # Encode categorical columns
        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])
        self.df['grf'] = le.fit_transform(self.df['grf'])

        # Split features and target
        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

        # Train-test split
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

        print("\n Best Parameters:", grid_search.best_params_)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Print evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        classification = classification_report(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        print("\n Accuracy:", accuracy)
        print("\n Confusion Matrix:\n", confusion)
        print("\n Classification Report:\n", classification)
        print("\n ROC AUC Score:", roc_auc)

        # Visualizations
        self.visualize_results(y_pred, y_proba)

    def visualize_results(self, y_pred, y_proba):
        # Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.figure(figsize=(6, 4))
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
            features = self.X.columns
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
            plt.title("Feature Importance")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.show()
        else:
            print(" Feature importance not available for this model.")


if __name__ == "__main__":
    model = XGBoostModel('ckd_prediction_dataset.csv')
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
