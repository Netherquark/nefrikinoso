import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SVMModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.model = None

    def preprocess_data(self):
        # Drop irrelevant columns
        self.df = self.df.drop(['affected', 'age_avg'], axis=1)

        # Encode categorical variables
        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])
        self.df['grf'] = le.fit_transform(self.df['grf'])

        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

    def train_test_split(self, test_size=0.2, random_state=42):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )

    def train_svm(self):
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }

        svm = SVC(probability=True, random_state=42)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)

        # Best parameters
        print("\nBest Parameters Found:")
        print(grid_search.best_params_)

        # Use the best estimator
        self.model = grid_search.best_estimator_

        # Get feature importance if linear kernel
        if grid_search.best_params_['kernel'] == 'linear':
            self.feature_importance()

    def feature_importance(self):
        # Coefficients represent importance for linear kernel
        coef = self.model.coef_.flatten()
        importance = pd.Series(coef, index=self.X.columns)
        importance = importance.sort_values(ascending=False)

        print("\n--- Feature Importance (Linear Kernel) ---")
        print(importance)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=importance.values, y=importance.index, palette='viridis')
        plt.title("Feature Importance (Linear Kernel)")
        plt.show()

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        classification = classification_report(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        print(f"\n--- Model Evaluation Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:\n", confusion)
        print("\nClassification Report:\n", classification)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        self.visualize_results(y_proba, confusion)

    def visualize_results(self, y_proba, confusion):
        print("\n--- Visualization of Results ---")

        # ✅ Confusion Matrix Fix
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        # Histogram of Predicted Probabilities
        plt.figure(figsize=(6, 4))
        sns.histplot(y_proba, bins=10, kde=True, color='skyblue')
        plt.title("Histogram of Predicted Probabilities")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.show()

        # Boxplot of Predicted Probabilities by True Class
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=self.y_test, y=y_proba)
        plt.title("Boxplot of Predicted Probabilities by True Class")
        plt.xlabel("True Class")
        plt.ylabel("Predicted Probability")
        plt.show()


if __name__ == "__main__":
    model = SVMModel('/Users/janhvidoijad/Desktop/Nefrikinoso/nefrikinoso/ckd_prediction_dataset.csv')
    model.preprocess_data()

    print("Training SVM Model...")
    model.train_test_split()
    model.train_svm()
    model.evaluate_model()
