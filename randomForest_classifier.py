import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

class CKDClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data['class'] = self.data['class'].map({'ckd': 1, 'notckd': 0})
        self.data.drop(columns=['affected', 'grf'], inplace=True, errors='ignore')
        self.X = self.data.drop(columns=['class'])
        self.y = self.data['class']
        self.X.fillna(self.X.mean(), inplace=True)

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        param_grid = {
            'n_estimators': [100, 200, 300],  
            'max_depth': [None, 10, 20],  
            'min_samples_split': [5, 10],  
            'min_samples_leaf': [2, 4],  
            'max_features': ['sqrt']  
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2, return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f'Best Model Accuracy: {acc * 100:.2f}%')
        print("Best Parameters:", grid_search.best_params_)
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        if hasattr(best_model, "predict_proba"):
            y_probs = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.show()

        self.plot_feature_importance(best_model)
        self.plot_feature_distribution()
        self.plot_training_validation_curve(grid_search.cv_results_)

    def plot_feature_importance(self, model):
        feature_importance = pd.Series(model.feature_importances_, index=self.X.columns).sort_values(ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.show()

    def plot_feature_distribution(self):
        plt.figure(figsize=(12, 8))
        self.X.boxplot(rot=90, patch_artist=True)
        plt.title('Feature Distribution (Detecting Outliers)')
        plt.xticks(rotation=90)
        plt.show()

    def plot_training_validation_curve(self, cv_results):
        train_scores = cv_results['mean_train_score']
        val_scores = cv_results['mean_test_score']

        plt.figure(figsize=(8, 5))
        plt.plot(range(len(train_scores)), train_scores, label='Training Accuracy', marker='o', color='blue')
        plt.plot(range(len(val_scores)), val_scores, label='Validation Accuracy', marker='o', color='red')
        plt.xlabel('Hyperparameter Combination')
        plt.ylabel('Accuracy Score')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
    ckd_model = CKDClassifier(file_path)
    ckd_model.load_data()
    ckd_model.train_and_evaluate()
