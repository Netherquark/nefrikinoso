import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class DecisionTreeCKD:
    def __init__(self, file_path, target_column, n_splits=5):
        self.file_path = file_path
        self.target_column = target_column
        self.n_splits = n_splits
        self.dataset = None
        self.results = []
        self.best_params = None

    def preprocess_data(self):
        self.dataset = pd.read_csv(self.file_path)
        self.dataset = self.dataset.drop(columns=["affected", "age_avg"], errors='ignore')
        self.dataset[self.target_column] = self.dataset[self.target_column].map({"ckd": 1, "notckd": 0})
        
        for col in self.dataset.columns:
            if self.dataset[col].dtype == "object":
                self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')
                self.dataset[col] = LabelEncoder().fit_transform(self.dataset[col].astype(str))
        
        self.dataset.dropna(inplace=True)
        print("[INFO] Data preprocessing complete.")
    
    def tune_hyperparameters(self, X, y):
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        print(f"[INFO] Best Hyperparameters: {self.best_params}")
    
    def evaluate_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        accuracy = np.trace(cm) / np.sum(cm)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return accuracy, sensitivity, specificity, precision, recall, f1
    
    def cross_validate(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        
        self.tune_hyperparameters(X, y)
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = DecisionTreeClassifier(**self.best_params, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.results.append(self.evaluate_metrics(y_test, y_pred))
        
        print("[INFO] Cross-validation complete.")
    
    def get_results(self):
        results_array = np.array(self.results)
        accuracy, sensitivity, specificity, precision, recall, f1 = results_array.mean(axis=0)
        return (f"[Decision Tree] Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, "
                f"Specificity: {specificity:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    def visualize_results(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.dataset[self.target_column])
        plt.title("CKD Class Distribution")
        plt.xlabel("CKD (1 = Yes, 0 = No)")
        plt.ylabel("Count")
        plt.show()
        
        model = DecisionTreeClassifier(**self.best_params, random_state=42)
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
        
        plt.figure(figsize=(15, 10))
        plot_tree(model, feature_names=X.columns, class_names=["Not CKD", "CKD"], filled=True)
        plt.title("Decision Tree Visualization")
        plt.show()
        
        print("[INFO] Visualizations generated.")

file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
target_column = "class"

dt_ckd = DecisionTreeCKD(file_path, target_column)
dt_ckd.preprocess_data()
dt_ckd.cross_validate()
print(dt_ckd.get_results())
dt_ckd.visualize_results()
