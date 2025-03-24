import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
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
        self.dataset[self.target_column] = self.dataset[self.target_column].map({"ckd": 1, "notckd": 0})
        self.X = self.dataset.drop(columns=[self.target_column, "grf", "stage", "affected", "age_avg"], errors='ignore')
        self.y = self.dataset[self.target_column]
    
    def cross_validate(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            
            model = GaussianNB()
            model.fit(X_train, y_train)
            self.models.append(model)
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) != 0 else 0
            specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) != 0 else 0
            
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
    
    def plot_results(self):
        results_array = np.array(self.results)
        print(f"[Naive Bayes] Accuracy: {np.mean(results_array[:, 0]):.4f}, "
              f"Sensitivity: {np.mean(results_array[:, 1]):.4f}, "
              f"Specificity: {np.mean(results_array[:, 2]):.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y, self.models[-1].predict(self.X))
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-CKD", "CKD"], yticklabels=["Non-CKD", "CKD"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        
        # ROC Curve
        plt.figure(figsize=(6, 4))
        for i in range(len(self.all_fpr)):
            plt.plot(self.all_fpr[i], self.all_tpr[i], alpha=0.3, label=f"Fold {i+1} (AUC = {self.all_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
        
        # Precision-Recall Curve
        plt.figure(figsize=(6, 4))
        for i in range(len(self.all_precision)):
            plt.plot(self.all_recall[i], self.all_precision[i], alpha=0.3, label=f"Fold {i+1}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()
        
        # Feature Importance (based on absolute log probabilities)
        feature_importance = np.abs(self.models[-1].theta_).mean(axis=0)
        feature_names = self.X.columns
        sorted_indices = np.argsort(feature_importance)[::-1]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feature_importance[sorted_indices], y=feature_names[sorted_indices], palette="viridis")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.show()
        
        # CKD vs Non-CKD Distribution
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
