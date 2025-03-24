import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

class CKDLogisticRegression:
    def __init__(self, dataset_path, target_column='class', k=5):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.k = k
        self.dataset = None
        self.model = LogisticRegression(solver='liblinear', C=1)
        self.scaler = StandardScaler()
    
    def load_data(self):
        self.dataset = pd.read_csv(self.dataset_path)
        
        # Remove redundant columns
        self.dataset.drop(columns=['affected', 'grf', 'age_avg'], inplace=True)
        
        # Convert target column to binary values
        self.dataset[self.target_column] = (self.dataset[self.target_column] == 'ckd').astype(int)
    
    def preprocess_data(self):
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def hyperparameter_tuning(self, X, y):
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=self.k, scoring='accuracy')
        grid_search.fit(X, y)
        self.model = LogisticRegression(solver='liblinear', C=grid_search.best_params_['C'])
    
    def train_evaluate(self):
        X, y = self.preprocess_data()
        self.hyperparameter_tuning(X, y)
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        accuracies, sensitivities, specificities = [], [], []
        all_y_test, all_y_pred, all_y_scores = [], [], []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_scores = self.model.predict_proba(X_test)[:, 1]
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            accuracies.append(accuracy)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_scores.extend(y_scores)
        
        mean_accuracy = np.mean(accuracies)
        mean_sensitivity = np.mean(sensitivities)
        mean_specificity = np.mean(specificities)
        
        print("\nModel Performance:")
        print("-----------------")
        print(f"Mean Accuracy: {mean_accuracy:.4f} ({mean_accuracy * 100:.2f}%)")
        print(f"Mean Sensitivity (Recall): {mean_sensitivity:.4f} ({mean_sensitivity * 100:.2f}%)")
        print(f"Mean Specificity: {mean_specificity:.4f} ({mean_specificity * 100:.2f}%)\n")
        
        self.visualize_results(np.array(all_y_test), np.array(all_y_pred), np.array(all_y_scores))
    
    def visualize_results(self, y_test, y_pred, y_scores):
        cm = confusion_matrix(y_test, y_pred)
        
        # Confusion Matrix Visualization
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
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