import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import json
import pickle

from predict_ckd import CKDPredictor
from XGBoost import XGBoostModel
from SVM import SVMModel
from Decision_tree import DecisionTreeModel
from LogisticRegression import LogisticRegressionModel
from knn import KNNModel
from NaiveBayes import NaiveBayesModel
from randomForest_classifier import RandomForestModel
from gradient_boost import GradientBoostModel
from catboost_ckd import CatBoostCKDModel
from stacked_ensemble import StackedEnsembleModel
from voting_ensemble import VotingEnsembleModel

from sklearn.metrics import classification_report, confusion_matrix

class CKDModelRunner:
    def __init__(self, dataset_path, roc_auc_path="roc_auc_values.json", best_model_path="best_model.pkl"):
        self.dataset_path = dataset_path
        self.roc_auc_path = roc_auc_path
        self.best_model_path = best_model_path
        self.reports = []
        self.results = []
        self.model_objects = {}
        self.roc_auc_values = {}
        self.results_df = pd.DataFrame()

        self.models_to_run = [
            (XGBoostModel, "XGBoost"),
            (SVMModel, "SVM"),
            (DecisionTreeModel, "Decision Tree"),
            (LogisticRegressionModel, "Logistic Regression"),
            (KNNModel, "K-Nearest Neighbors"),
            (NaiveBayesModel, "Naive Bayes"),
            (RandomForestModel, "Random Forest"),
            (GradientBoostModel, "Gradient Boosting"),
            (CatBoostCKDModel, "CatBoost"),
            (StackedEnsembleModel, "Stacked Ensemble Learning"),
            (VotingEnsembleModel, "Voting")
        ]

    def run_all_models(self):
        for model_class, model_name in self.models_to_run:
            print(f"\n Running {model_name}...")
            try:
                model = model_class(self.dataset_path)
                model.preprocess_data()
                model.train_test_split()

                start_time = time.time()
                model.train_model()
                training_time = time.time() - start_time

                scores = model.evaluate_model(return_scores=True)
                scores['Model'] = model_name
                scores['Training_Time'] = training_time

                y_true, y_pred = model.y_test, model.model.predict(model.X_test)
                scores['y_true'] = y_true
                scores['y_pred'] = y_pred

                self.model_objects[model_name] = model.model

                if hasattr(model.model, 'feature_importances_'):
                    importances = model.model.feature_importances_
                    scores['Feature_Importances'] = pd.Series(importances, index=model.X.columns).sort_values(ascending=False)
                elif hasattr(model.model, 'coef_'):
                    importances = abs(model.model.coef_[0])
                    scores['Feature_Importances'] = pd.Series(importances, index=model.X.columns).sort_values(ascending=False)

                self.reports.append(scores)
                self.results.append({
                    "Model": model_name,
                    "Accuracy": scores["Accuracy"],
                    "ROC_AUC": scores["ROC_AUC"],
                    "Training_Time": training_time
                })
                self.roc_auc_values[model_name] = scores["ROC_AUC"]
                self.feature_order = list(model.X.columns)

            except Exception as e:
                print(f" Error in {model_name}: {e}")

        # Build the DataFrame
        self.results_df = pd.DataFrame(self.results)

        # Normalize the ROC column name
        if 'roc_auc' in self.results_df.columns:
            self.results_df.rename(columns={'roc_auc':'ROC_AUC'}, inplace=True)

        # Debugging: print normalized columns
        print("\nFinal Comparison Table (normalized):")
        print(self.results_df.sort_values(by="Accuracy", ascending=False))
        print("Columns now:", self.results_df.columns.tolist())

    def save_roc_auc(self):
        with open(self.roc_auc_path, 'w') as f:
            json.dump(self.roc_auc_values, f, indent=4)
        print(f"\nROC AUC values saved to '{self.roc_auc_path}'")

    def plot_metrics(self):
        self.plot_metric_bar("Accuracy", "crest")
        self.plot_metric_bar("ROC_AUC", "viridis")
        self.plot_metric_bar("Training_Time", "magma")
        self.plot_classification_report()
        self.plot_confusion_matrices()

    def plot_metric_bar(self, metric, palette):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metric, y="Model", data=self.results_df.sort_values(by=metric, ascending=False), palette=palette)
        plt.title(f"{metric} Comparison")
        plt.xlabel(metric)
        plt.ylabel("Model")
        plt.tight_layout()
        plt.show()

    def plot_classification_report(self):
        rows = []
        for report in self.reports:
            clf_report = classification_report(report['y_true'], report['y_pred'], output_dict=True)
            for label, metrics in clf_report.items():
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    rows.append({
                        "Model": report["Model"],
                        "Class": label,
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1-Score": metrics["f1-score"]
                    })

        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        sns.barplot(data=df, x="F1-Score", y="Model", hue="Class", ax=axes[0])
        axes[0].set_title("F1-Score Comparison by Class")

        sns.barplot(data=df, x="Precision", y="Model", hue="Class", ax=axes[1])
        axes[1].set_title("Precision Comparison by Class")

        sns.barplot(data=df, x="Recall", y="Model", hue="Class", ax=axes[2])
        axes[2].set_title("Recall Comparison by Class")

        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend(title='Class', loc='lower right')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self):
        n_models = len(self.reports)
        cols = 3
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

        for i, report in enumerate(self.reports):
            cm = confusion_matrix(report['y_true'], report['y_pred'])
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f"{report['Model']}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        for j in range(i + 1, rows * cols):
            ax = axes[j // cols, j % cols] if rows > 1 else axes[j]
            ax.axis('off')

        plt.suptitle("Confusion Matrices", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.show()

    def get_best_model(self):
        best_row = self.results_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]
        best_model_name = best_row["Model"]
        best_model = self.model_objects[best_model_name]
        print(f"\nBest model based on ROC AUC: {best_model_name}")
        return best_model_name, best_model

    def predict_from_user_input(self, model):
        predictor = CKDPredictor(model, self.feature_order)
        user_data = predictor.get_user_input()
        prediction = predictor.predict(user_data)
        print(f"\nPrediction Result: The patient is predicted to have **{prediction}**.")
        
    def predict_from_user_input_gui(self, user_input_dict):
        self.run_all_models()
        self.save_roc_auc()
        model, best_model = self.get_best_model()
        predictor = CKDPredictor(best_model, self.feature_order)
        prediction = predictor.predict_from_input_dict(user_input_dict)
        return prediction


    def run(self):
        self.run_all_models()
        self.save_roc_auc()
        self.plot_metrics()
        best_model_name, best_model = self.get_best_model()
        self.predict_from_user_input(best_model)

if __name__ == "__main__":
    runner = CKDModelRunner(dataset_path="ckd_prediction_dataset.csv")
    runner.run()
