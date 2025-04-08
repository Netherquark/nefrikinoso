import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

class CKDClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.best_model = None
        self.y_pred = None
        self.y_proba = None

    def load_data(self):
        """Loads and preprocesses the CKD dataset."""
        self.data = pd.read_csv(self.file_path)
        self.data['class'] = self.data['class'].map({'ckd': 1, 'notckd': 0})
        self.data.drop(columns=['affected', 'grf'], inplace=True, errors='ignore')

        self.X = self.data.drop(columns=['class'])
        self.y = self.data['class']
        self.X.fillna(self.X.mean(), inplace=True)

        # SMOTE for class balancing
        smote = SMOTE(sampling_strategy=0.8, random_state=42)
        self.X, self.y = smote.fit_resample(self.X, self.y)

    def train_and_evaluate(self):
        """Trains and evaluates the RandomForest model with hyperparameter tuning."""
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt']
        }

        rf = RandomForestClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.y_pred = self.best_model.predict(self.X_test)
        self.y_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        acc_train = grid_search.best_score_
        acc_test = accuracy_score(self.y_test, self.y_pred)

        print(f'Training Accuracy (CV): {acc_train * 100:.2f}%')
        print(f'Test Accuracy: {acc_test * 100:.2f}%')
        print("Best Parameters:", grid_search.best_params_)
        print("Classification Report:\n", classification_report(self.y_test, self.y_pred))

        self.visualize_results()

    def visualize_results(self):
        """Visualizes confusion matrix, ROC curve, and feature importance."""
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

        # Feature Importance
        importances = self.best_model.feature_importances_
        features = self.X.columns
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
    ckd_model = CKDClassifier(file_path)
    ckd_model.load_data()
    ckd_model.train_and_evaluate()
