import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

class CKD_KNN_CrossValidator:
    def __init__(self, file_path, target_column="class", k_neighbors=9, n_splits=5):
        self.file_path = file_path
        self.dataset = None
        self.target_column = target_column
        self.k_neighbors = k_neighbors
        self.n_splits = n_splits
        self.statistics = []
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """Loads the dataset and preprocesses categorical variables."""
        self.dataset = pd.read_csv(self.file_path)
        
        # Convert target variable ('class') into binary values (0 = notckd, 1 = ckd)
        self.dataset[self.target_column] = self.dataset[self.target_column].map({'ckd': 1, 'notckd': 0})

        # Convert object columns to numeric if necessary
        self.dataset = self.dataset.apply(pd.to_numeric, errors='coerce')

        # Fill missing values with median (better for robustness)
        self.dataset.fillna(self.dataset.median(), inplace=True)

        print("Preprocessing Complete!")
        print(self.dataset.info())
        print("\nSummary Statistics:\n", self.dataset.describe())

    def visualize_data(self):
        """Displays basic visualizations of the dataset."""
        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.dataset[self.target_column])
        plt.title("Class Distribution (CKD vs Non-CKD)")
        plt.show()

        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.dataset.corr(), annot=False, cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def evaluate_model(self):
        """Performs KNN classification with cross-validation."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            accuracy = np.trace(cm) / np.sum(cm)
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0

            self.statistics.append([accuracy, sensitivity, specificity])

        print("\nModel Evaluation Complete!")

    def get_results(self):
        """Returns the averaged evaluation metrics."""
        stats_df = pd.DataFrame(self.statistics, columns=["Accuracy", "Sensitivity", "Specificity"])
        print("\nCross-Validation Results:\n", stats_df)
        
        mean_accuracy = stats_df["Accuracy"].mean()
        mean_sensitivity = stats_df["Sensitivity"].mean()
        mean_specificity = stats_df["Specificity"].mean()

        print("\n[K-NN] Accuracy: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}".format(
            mean_accuracy, mean_sensitivity, mean_specificity
        ))

        return mean_accuracy, mean_sensitivity, mean_specificity

    def plot_confusion_matrix(self):
        """Plots the confusion matrix from a final KNN model."""
        if not self.statistics:
            print("No confusion matrix available. Run evaluate_model() first.")
            return

        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        model.fit(X, y)
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not CKD", "CKD"], yticklabels=["Not CKD", "CKD"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

# Usage
file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
ckd_knn_validator = CKD_KNN_CrossValidator(file_path)
ckd_knn_validator.visualize_data()
ckd_knn_validator.evaluate_model()
ckd_knn_validator.get_results()
ckd_knn_validator.plot_confusion_matrix()

