import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

class CKDClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """Loads and preprocesses the CKD dataset."""
        self.data = pd.read_csv(self.file_path)
        self.data['class'] = self.data['class'].map({'ckd': 1, 'notckd': 0})
        self.data.drop(columns=['affected', 'grf'], inplace=True, errors='ignore')
        
        # Splitting features and labels
        self.X = self.data.drop(columns=['class'])
        self.y = self.data['class']
        
        # Handling missing values
        self.X.fillna(self.X.mean(), inplace=True)

        # Apply SMOTE for balancing the dataset
        smote = SMOTE(sampling_strategy=0.8, random_state=42)  # Less aggressive oversampling
        self.X, self.y = smote.fit_resample(self.X, self.y)

    def train_and_evaluate(self):
        """Trains and evaluates the RandomForest model with hyperparameter tuning."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        param_grid = {
            'n_estimators': [50, 100, 150],  # Fewer trees for better generalization
            'max_depth': [5, 10, 15],  # Control tree depth
            'min_samples_split': [10, 20],  # Prevent over-splitting
            'min_samples_leaf': [5, 10],  # Ensure leaves have enough data
            'max_features': ['sqrt']  
        }

        rf = RandomForestClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Better cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2, return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc_train = grid_search.best_score_
        acc_test = accuracy_score(y_test, y_pred)

        print(f'Training Accuracy (CV): {acc_train * 100:.2f}%')
        print(f'Test Accuracy: {acc_test * 100:.2f}%')
        print("Best Parameters:", grid_search.best_params_)
        print(classification_report(y_test, y_pred))

        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(best_model, X_test, y_test)
        self.plot_feature_importance(best_model)
        self.plot_training_validation_curve(grid_search.cv_results_)

    def plot_confusion_matrix(self, y_test, y_pred):
        """Plots the confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, model, X_test, y_test):
        """Plots the ROC curve."""
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
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

    def plot_feature_importance(self, model):
        """Plots feature importance based on the trained model."""
        feature_importance = pd.Series(model.feature_importances_, index=self.X.columns).sort_values(ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.show()

    def plot_training_validation_curve(self, cv_results):
        """Plots training vs validation accuracy."""
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
