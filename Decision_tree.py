import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class DecisionTreeCV:
    def __init__(self, file_path, target_column, n_splits=5):
        self.file_path = file_path
        self.target_column = target_column
        self.n_splits = n_splits
        self.dataset = None
        self.results = []

    def preprocess_data(self):
        """Load and preprocess the dataset (handle missing values, categorical encoding, etc.)."""
        # Load dataset
        self.dataset = pd.read_csv(self.file_path)

        # Convert 'class' column to binary (1 for CKD, 0 for notCKD)
        self.dataset[self.target_column] = self.dataset[self.target_column].map({"ckd": 1, "notckd": 0})

        # Convert categorical columns to numeric
        for col in self.dataset.columns:
            if self.dataset[col].dtype == "object":  # If column is non-numeric
                try:
                    self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')  # Convert if possible
                except:
                    self.dataset[col] = LabelEncoder().fit_transform(self.dataset[col].astype(str))  # Otherwise, encode

        # Drop rows with missing values
        self.dataset = self.dataset.dropna()

        print("[INFO] Data preprocessing complete.")
    
    def evaluate_metrics(self, y_true, y_pred):
        """Calculate performance metrics based on confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        accuracy = np.trace(cm) / np.sum(cm)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return accuracy, sensitivity, specificity, precision, recall, f1

    def cross_validate(self):
        """Perform k-fold cross-validation on the decision tree model."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            self.results.append(self.evaluate_metrics(y_test, y_pred))

        print("[INFO] Cross-validation complete.")

    def get_results(self):
        """Return averaged cross-validation results."""
        results_array = np.array(self.results)
        accuracy, sensitivity, specificity, precision, recall, f1 = results_array.mean(axis=0)
        return (f"[Decision Tree] Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, "
                f"Specificity: {specificity:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    def visualize_results(self):
        """Generate visualizations for class distribution and feature importance."""
        # 1. Class Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.dataset[self.target_column])
        plt.title("CKD Class Distribution")
        plt.xlabel("CKD (1 = Yes, 0 = No)")
        plt.ylabel("Count")
        plt.show()

        # 2. Feature Importance
        model = DecisionTreeClassifier(random_state=42)
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        model.fit(X, y)

        importances = model.feature_importances_
        features = X.columns

        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances, y=features)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()

        print("[INFO] Visualizations generated.")

# ========== RUN MODEL ==========

# File path to dataset
file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
target_column = "class"

# Initialize the DecisionTreeCV class
dt_cv = DecisionTreeCV(file_path, target_column)

# Step 1: Preprocess Data
dt_cv.preprocess_data()

# Step 2: Train Model with Cross-Validation
dt_cv.cross_validate()

# Step 3: Print Evaluation Results
print(dt_cv.get_results())

# Step 4: Generate Visualizations
dt_cv.visualize_results()
