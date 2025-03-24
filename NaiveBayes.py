import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve

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
        class_mapping = {"ckd": 1, "notckd": 0, "ckd.": 1, "not ckd": 0}  # Handling minor variations
        self.dataset[self.target_column] = self.dataset[self.target_column].map(class_mapping)

        # Drop unnecessary or missing columns safely
        cols_to_drop = [col for col in ["grf", "affected", "age_avg"] if col in self.dataset.columns]
        self.X = self.dataset.drop(columns=cols_to_drop + [self.target_column])

        # Handle missing values
        self.X = self.X.fillna(self.X.mean())  # Fill NaNs with mean values
        self.y = self.dataset[self.target_column].dropna()  # Remove NaNs from target column
        self.X = self.X.loc[self.y.index]  # Ensure X and y have matching indices

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

            self.all_fpr.append(fpr)
            self.all_tpr.append(tpr)
            self.all_auc.append(roc_auc)
            self.all_precision.append(precision)
            self.all_recall.append(recall)

            self.results.append([accuracy, sensitivity, specificity])

    def hyperparameter_tuning(self):
        param_grid = {'var_smoothing': np.logspace(-9, 0, 10)}
        grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X, self.y)
        return grid_search.best_params_['var_smoothing']

    def plot_results(self):
        results_array = np.array(self.results)
        print(f"[Naive Bayes] Accuracy: {np.mean(results_array[:, 0]):.4f}, "
              f"Sensitivity: {np.mean(results_array[:, 1]):.4f}, "
              f"Specificity: {np.mean(results_array[:, 2]):.4f}")

        cm = confusion_matrix(self.y, self.models[-1].predict(self.X))
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-CKD", "CKD"], yticklabels=["Non-CKD", "CKD"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        plt.figure(figsize=(6, 4))
        for i in range(len(self.all_fpr)):
            plt.plot(self.all_fpr[i], self.all_tpr[i], alpha=0.3, label=f"Fold {i+1} (AUC = {self.all_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

        plt.figure(figsize=(6, 4))
        for i in range(len(self.all_precision)):
            plt.plot(self.all_recall[i], self.all_precision[i], alpha=0.3, label=f"Fold {i+1}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()

        plt.figure(figsize=(5, 4))
        sns.countplot(x=self.y, palette="coolwarm")
        plt.xticks(ticks=[0, 1], labels=["Non-CKD", "CKD"])
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("CKD vs Non-CKD Distribution")
        plt.show()


# Load dataset
df = pd.read_csv("/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv")

# Run the model
nb_ckd = NaiveBayesCKD(dataset=df, target_column="class")
nb_ckd.preprocess_data()
nb_ckd.cross_validate()
nb_ckd.plot_results()
