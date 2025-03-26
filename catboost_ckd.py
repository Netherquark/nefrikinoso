import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class CatBoostModel:
    def __init__(self, file_path, params=None):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        if params is None:
            params = {'iterations': 1000, 'depth': 6, 'learning_rate': 0.1, 'loss_function': 'Logloss', 'verbose': 100}
        self.model = CatBoostClassifier(**params)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self):
        self.df = self.df.drop(['affected', 'age_avg'], axis=1)
        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])
        self.df['grf'] = le.fit_transform(self.df['grf'])
        X = self.df.drop('class', axis=1)
        y = self.df['class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def hyperparameter_tuning(self):
        param_dist = {
            'iterations': [500, 1000, 1500],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
        random_search = RandomizedSearchCV(CatBoostClassifier(loss_function='Logloss', verbose=0), param_distributions=param_dist, n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
        random_search.fit(self.X_train, self.y_train)
        print("Best parameters found: ", random_search.best_params_)
        self.model = CatBoostClassifier(**random_search.best_params_, loss_function='Logloss', verbose=100)

    def train(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Please preprocess data using preprocess_data() method before training.")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not loaded. Please preprocess data using preprocess_data() method before evaluation.")
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        classification = classification_report(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        print("Accuracy:", accuracy)
        print("\nConfusion Matrix:\n", confusion)
        print("\nClassification Report:\n", classification)
        print("\nROC AUC Score:", roc_auc)
        
        self.plot_confusion_matrix(confusion)
        self.plot_roc_curve(y_proba)
        self.plot_feature_importance()
        self.plot_precision_recall_curve(y_proba)
        self.plot_prediction_distribution(y_pred)

    def plot_feature_importance(self):
        importance = self.model.get_feature_importance()
        features = self.df.drop('class', axis=1).columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features, palette='viridis')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()
    
    def plot_confusion_matrix(self, confusion):
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_roc_curve(self, y_proba):
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(self.y_test, y_proba):.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    
    def plot_precision_recall_curve(self, y_proba):
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
    
    def plot_prediction_distribution(self, y_pred):
        plt.figure(figsize=(6, 4))
        sns.histplot(y_pred, bins=2, kde=False, color='purple')
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.title('Prediction Distribution')
        plt.show()

if __name__ == "__main__":
    model = CatBoostModel("ckd_prediction_dataset.csv")
    model.preprocess_data()
    model.hyperparameter_tuning()
    model.train()
    model.evaluate()
