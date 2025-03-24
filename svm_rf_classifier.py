import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class CKDClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
        # Convert target variable to binary format
        self.data['class'] = self.data['class'].map({'ckd': 1, 'notckd': 0})
        
        # Convert 'grf' to numeric
        self.data['grf'] = pd.to_numeric(self.data['grf'], errors='coerce')
        
        # Drop redundant columns
        self.data.drop(columns=['age_avg'], inplace=True)
        
        # Split features and target
        self.X = self.data.drop(columns=['class'])
        self.y = self.data['class']
        
        # Handle missing values
        self.X.fillna(self.X.mean(), inplace=True)
    
    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        models = {
            'SVM (RBF Kernel)': SVC(kernel='rbf', C=1, gamma='scale'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f'Accuracy of {name}: {acc * 100:.2f}%')
            print(classification_report(y_test, y_pred))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {name}')
            plt.show()

if __name__ == "__main__":
    file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
    ckd_model = CKDClassifier(file_path)
    ckd_model.load_data()
    ckd_model.train_and_evaluate()
