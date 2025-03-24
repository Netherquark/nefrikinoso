import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

class CKDLogisticRegression:
    def __init__(self, dataset_path, target_column='affected', k=5):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.k = k
        self.dataset = None
        self.model = LogisticRegression(solver='liblinear')
        self.scaler = StandardScaler()
    
    def load_data(self):
        self.dataset = pd.read_csv(self.dataset_path)
        
        # Convert target column to integer
        self.dataset[self.target_column] = self.dataset[self.target_column].astype(int)
        
        # Convert 'grf' column to numeric and handle missing values
        self.dataset['grf'] = pd.to_numeric(self.dataset['grf'], errors='coerce')
        self.dataset['grf'].fillna(self.dataset['grf'].median(), inplace=True)
        
        # Drop redundant columns
        if 'class' in self.dataset.columns:
            self.dataset.drop(columns=['class'], inplace=True)
        if 'age_avg' in self.dataset.columns:
            self.dataset.drop(columns=['age_avg'], inplace=True)
    
    def preprocess_data(self):
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def train_evaluate(self):
        X, y = self.preprocess_data()
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        accuracies, sensitivities, specificities = [], [], []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            accuracies.append(accuracy)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
        print(f"Mean Sensitivity (Recall): {np.mean(sensitivities):.4f}")
        print(f"Mean Specificity: {np.mean(specificities):.4f}")
        
        self.visualize_results(y_test, y_pred, cm)
    
    def visualize_results(self, y_test, y_pred, cm):
        # Confusion Matrix Visualization
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Feature Importance Visualization
        feature_importance = pd.DataFrame({'Feature': self.dataset.drop(columns=[self.target_column]).columns,
                                           'Coefficient': self.model.coef_[0]})
        feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='coolwarm')
        plt.title("Feature Importance in Logistic Regression")
        plt.show()
        
        print("Classification Report:\n", classification_report(y_test, y_pred))

# Usage Example
dataset_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
ckd_model = CKDLogisticRegression(dataset_path)
ckd_model.load_data()
ckd_model.train_evaluate()

