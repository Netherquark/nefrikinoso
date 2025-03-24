import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

class CKD_KNN_CrossValidator:
    def __init__(self, file_path, target_column="class", n_splits=5):
        self.file_path = file_path
        self.dataset = None
        self.target_column = target_column
        self.n_splits = n_splits
        self.best_k = None
        self.statistics = []
        self.load_and_preprocess_data()
    
    def load_and_preprocess_data(self):
        self.dataset = pd.read_csv(self.file_path)
        self.dataset.drop(columns=["affected", "grf"], inplace=True)
        self.dataset[self.target_column] = self.dataset[self.target_column].map({'ckd': 1, 'notckd': 0})
        self.dataset = self.dataset.apply(pd.to_numeric, errors='coerce')
        self.dataset.fillna(self.dataset.median(), inplace=True)
        scaler = StandardScaler()
        features = self.dataset.drop(columns=[self.target_column])
        self.dataset[features.columns] = scaler.fit_transform(features)
    
    def visualize_data(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.dataset[self.target_column])
        plt.title("Class Distribution (CKD vs Non-CKD)")
        plt.show()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.dataset.corr(), annot=False, cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.show()
        
        self.dataset.hist(figsize=(12, 10), bins=20, edgecolor='black')
        plt.suptitle("Feature Distributions")
        plt.show()
    
    def tune_hyperparameters(self):
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        param_grid = {'n_neighbors': range(1, 20, 2)}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)
        self.best_k = grid_search.best_params_['n_neighbors']
        print(f"Best k: {self.best_k}")
    
    def evaluate_model(self):
        if self.best_k is None:
            self.tune_hyperparameters()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model = KNeighborsClassifier(n_neighbors=self.best_k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            accuracy = np.trace(cm) / np.sum(cm)
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
            self.statistics.append([accuracy, sensitivity, specificity])
    
    def get_results(self):
        stats_df = pd.DataFrame(self.statistics, columns=["Accuracy", "Sensitivity", "Specificity"])
        mean_accuracy = stats_df["Accuracy"].mean()
        mean_sensitivity = stats_df["Sensitivity"].mean()
        mean_specificity = stats_df["Specificity"].mean()
        print(f"[K-NN] Accuracy: {mean_accuracy:.4f}, Sensitivity: {mean_sensitivity:.4f}, Specificity: {mean_specificity:.4f}")
        return mean_accuracy, mean_sensitivity, mean_specificity
    
    def plot_confusion_matrix(self):
        if not self.statistics:
            print("No confusion matrix available. Run evaluate_model() first.")
            return
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        model = KNeighborsClassifier(n_neighbors=self.best_k)
        model.fit(X, y)
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not CKD", "CKD"], yticklabels=["Not CKD", "CKD"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
ckd_knn_validator = CKD_KNN_CrossValidator(file_path)
ckd_knn_validator.visualize_data()
ckd_knn_validator.evaluate_model()
ckd_knn_validator.get_results()
ckd_knn_validator.plot_confusion_matrix()
