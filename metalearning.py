import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import individual model wrappers
from XGBoost import XGBoostModel
from randomForest_classifier import RandomForestModel
from gradient_boost import GradientBoostModel
from SVM import SVMModel
from catboost_ckd import CatBoostCKDModel
from LogisticRegression import LogisticRegressionModel
from Decision_tree import DecisionTreeModel
from knn import KNNModel
from NaiveBayes import NaiveBayesModel
from ensemble_learning import EnsembleLearningModel

class MetaLearningModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.meta_model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self):
        df = self.df.copy()
        df.drop(columns=['affected', 'age_avg'], inplace=True, errors='ignore')

        le = LabelEncoder()
        if df['class'].dtype == 'object':
            df['class'] = le.fit_transform(df['class'])
        if 'grf' in df.columns and df['grf'].dtype == 'object':
            df['grf'] = le.fit_transform(df['grf'])

        self.X = df.drop(columns=['class'])
        self.y = df['class']

    def train_test_split(self):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )

    def train_model(self):
        model_classes = [
            (XGBoostModel, "XGBoost"),
            (RandomForestModel, "Random Forest"),
            (GradientBoostModel, "Gradient Boost"),
            (SVMModel, "SVM"),
            (CatBoostCKDModel, "CatBoost"),
            (LogisticRegressionModel, "Logistic Regression"),
            (DecisionTreeModel, "Decision Tree"),
            (KNNModel, "KNN"),
            (NaiveBayesModel, "Naive Bayes"),
            (EnsembleLearningModel, "Ensemble Learning")
        ]

        print("🔍 Evaluating all base models...")
        model_scores = []
        for model_class, name in model_classes:
            try:
                model = model_class(self.dataset_path)
                model.preprocess_data()
                model.train_test_split()
                model.train_model()
                scores = model.evaluate_model(return_scores=True)
                model_scores.append((name, model, scores["Accuracy"], scores["ROC_AUC"]))
            except Exception as e:
                print(f"⚠️ Skipping {name} due to error: {e}")

        # Select top 3 models by Accuracy
        top_models = sorted(model_scores, key=lambda x: x[2], reverse=True)[:3]

        print("\n✅ Top 3 models selected for meta-learning:")
        for name, _, acc, auc in top_models:
            print(f"{name}: Accuracy={acc:.4f}, ROC_AUC={auc:.4f}")

        estimators = [(name.lower().replace(" ", "_"), model.model) for name, model, _, _ in top_models]
        self.meta_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            passthrough=True,
            cv=5
        )

        self.meta_model.fit(self.X_train, self.y_train)
        self.model = self.meta_model

    def evaluate_model(self, return_scores=False):
        y_pred = self.meta_model.predict(self.X_test)
        y_proba = self.meta_model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        if return_scores:
            return {
                "Accuracy": accuracy,
                "ROC_AUC": roc_auc,
                "y_true": self.y_test,
                "y_pred": y_pred
            }

        print("\n📊 Meta-Model Evaluation")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.4f}")

        self.visualize_results(y_pred)

    def visualize_results(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix - Meta Model")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
