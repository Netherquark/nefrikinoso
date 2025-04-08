import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)

class NaiveBayesCKD:
    def __init__(self, dataset, target_column, n_splits=5):
        self.dataset = dataset
        self.target_column = target_column
        self.n_splits = n_splits
        self.results = []
        self.models = []
        self.all_fpr, self.all_tpr, self.all_auc = [], [], []
        self.all_precision, self.all_recall = [], []

    def preprocess_data(self):
        # Standardize class labels
        self.dataset[self.target_column] = self.dataset[self.target_column].str.lower().str.strip()
        class_mapping = {"ckd": 1, "notckd": 0, "ckd.": 1, "not ckd": 0}
        self.dataset[self.target_column] = self.dataset[self.target_column].map(class_mapping)

        # Drop unnecessary columns if they exist
        cols_to_drop = [col for col in ["grf", "affected", "age_avg"] if col in self.dataset.columns]
        self.X = self.dataset.drop(columns=cols_to_drop + [self.target_column])

        # Handle missing values
        self.X = self.X.fillna(self.X.mean(numeric_only=True))
        self.y = self.dataset[self.target_column].dropna()
        self.X = self.X.loc[self.y.index]

    def hyperparameter_tuning(self):
        param_grid = {'var_smoothing': np.logspace(-9, 0, 10)}
        grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X, self.y)
        return grid_search.best_params_['var_smoothing']

    def cross_validate(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        best_var_smoothing = self.hyperparameter_tuning()

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            model = GaussianNB(var_smoothing=best_var_smoothing)
            model.fit(X_train, y_train)
            self.models.append(model)

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            self.results.append([accuracy, sensitivity, specificity])
            self.all_fpr.append(fpr)
            self.all_tpr.append(tpr)
            self.all_auc.append(roc_auc)
            self.all_precision.append(precision)
            self.all_recall.append(recall)

    def plot_results(self):
        accuracies, sensitivities, specificities = zip(*self.results)

        print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
        print(f"Average Sensitivity (Recall): {np.mean(sensitivities):.4f}")
        print(f"Average Specificity: {np.mean(specificities):.4f}")

        # ROC Curves
        plt.figure(figsize=(8, 6))
        for i in range(len(self.all_fpr)):
            plt.plot(self.all_fpr[i], self.all_tpr[i], label=f"Fold {i+1} AUC = {self.all_auc[i]:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (Cross-Validation)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Confusion matrix from the last fold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        _, last_test_index = list(kf.split(self.X))[-1]
        X_test_last = self.X.iloc[last_test_index]
        y_test_last = self.y.iloc[last_test_index]
        last_model = self.models[-1]
        y_pred_last = last_model.predict(X_test_last)

        print("\nClassification Report (Last Fold):")
        print(classification_report(y_test_last, y_pred_last))

        cm = confusion_matrix(y_test_last, y_pred_last)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (Last Fold)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

# -------------------------
# 🚀 Load and Run Everything
# -------------------------

# Load dataset
df = pd.read_csv("/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv")

# Run the model
nb_ckd = NaiveBayesCKD(dataset=df, target_column="class")
nb_ckd.preprocess_data()
nb_ckd.cross_validate()
nb_ckd.plot_results()
