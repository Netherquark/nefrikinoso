import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import json

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
from voting_ensemble import CKDEnsembleModel

from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "ckd_prediction_dataset.csv"
# Updated constant: saving ROC AUC values instead of recall values
ROC_AUC_OUTPUT_PATH = "roc_auc_values.json"

def run_and_collect(model_class, model_name):
    model = model_class(DATASET_PATH)

    model.preprocess_data()
    model.train_test_split()

    start_time = time.time()
    model.train_model()
    end_time = time.time()

    training_time = end_time - start_time
    scores = model.evaluate_model(return_scores=True)
    scores['Model'] = model_name
    scores['Training_Time'] = training_time

    # Predictions
    y_true, y_pred = model.y_test, model.model.predict(model.X_test)
    scores['y_true'] = y_true
    scores['y_pred'] = y_pred

    # Feature Importances
    try:
        importances = None
        if hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
        elif hasattr(model.model, 'coef_'):
            importances = abs(model.model.coef_[0])
        if importances is not None:
            feature_names = model.X.columns
            scores['Feature_Importances'] = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    except:
        pass

    return scores

def plot_classification_report_dashboard(reports):
    rows = []
    for report in reports:
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

def plot_confusion_matrices(reports):
    n_models = len(reports)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    for i, report in enumerate(reports):
        cm = confusion_matrix(report['y_true'], report['y_pred'])
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"{report['Model']}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

    # Turn off any unused axes
    for j in range(i + 1, rows * cols):
        ax = axes[j // cols, j % cols] if rows > 1 else axes[j]
        ax.axis('off')

    plt.suptitle("Confusion Matrices", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.show()


def plot_metric_bar(dataframe, metric, palette):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric, y="Model", data=dataframe.sort_values(by=metric, ascending=False), palette=palette)
    plt.title(f"{metric} Comparison")
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()

def plot_all_top_features(reports, top_n=5):
    combined_features = []

    for report in reports:
        if "Feature_Importances" in report:
            top_features = report["Feature_Importances"].head(top_n)
            for feature, importance in top_features.items():
                combined_features.append({
                    "Model": report["Model"],
                    "Feature": feature,
                    "Importance": importance
                })

    df = pd.DataFrame(combined_features)
    plt.figure(figsize=(12, 7))
    sns.barplot(x="Importance", y="Feature", hue="Model", data=df, orient='h')
    plt.title(f"Top {top_n} Features Across Models")
    plt.tight_layout()
    plt.show()

def main():
    results = []
    detailed_reports = []
    model_objects = {}
    # Replace recall_values with roc_auc_values dictionary
    roc_auc_values = {}

    models_to_run = [
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
        (CKDEnsembleModel, "Voting")
    ]

    for model_class, model_name in models_to_run:
        print(f"\n Running {model_name}...")
        try:
            model = model_class(DATASET_PATH)
            model.preprocess_data()
            model.train_test_split()
            start_time = time.time()
            model.train_model()
            end_time = time.time()

            training_time = end_time - start_time
            scores = model.evaluate_model(return_scores=True)
            scores['Model'] = model_name
            scores['Training_Time'] = training_time

            y_true, y_pred = model.y_test, model.model.predict(model.X_test)
            scores['y_true'] = y_true
            scores['y_pred'] = y_pred

            # Store model for later use
            model_objects[model_name] = model.model

            if hasattr(model.model, 'feature_importances_'):
                importances = model.model.feature_importances_
                scores['Feature_Importances'] = pd.Series(importances, index=model.X.columns).sort_values(ascending=False)
            elif hasattr(model.model, 'coef_'):
                importances = abs(model.model.coef_[0])
                scores['Feature_Importances'] = pd.Series(importances, index=model.X.columns).sort_values(ascending=False)

            detailed_reports.append(scores)
            results.append({
                "Model": model_name,
                "Accuracy": scores["Accuracy"],
                "ROC_AUC": scores["ROC_AUC"],
                "Training_Time": training_time
            })

            # Instead of extracting recall values, extract and store the ROC AUC score
            roc_auc_values[model_name] = scores["ROC_AUC"]

        except Exception as e:
            print(f" Error in {model_name}: {e}")

    results_df = pd.DataFrame(results)

    print("\n Final Comparison Table:")
    print(results_df.sort_values(by="Accuracy", ascending=False))

    # Save ROC AUC values to JSON
    with open(ROC_AUC_OUTPUT_PATH, 'w') as f:
        json.dump(roc_auc_values, f, indent=4)

    print(f"\nROC AUC values for each model saved to '{ROC_AUC_OUTPUT_PATH}'")

    # Metric Charts
    plot_metric_bar(results_df, "Accuracy", "crest")
    plot_metric_bar(results_df, "ROC_AUC", "viridis")
    plot_metric_bar(results_df, "Training_Time", "magma")
    plot_classification_report_dashboard(detailed_reports)
    plot_confusion_matrices(detailed_reports)
    plot_all_top_features(detailed_reports, top_n=5)

    # Get the best model (by ROC_AUC)
    best_model_row = results_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]
    best_model_name = best_model_row["Model"]
    best_model = model_objects[best_model_name]

    print(f"\nUsing best model '{best_model_name}' for CKD prediction from user input...")

    # User input and CKD prediction
    feature_order = list(model.X.columns)
    predictor = CKDPredictor(best_model, feature_order)

    user_data = predictor.get_user_input()
    prediction = predictor.predict(user_data)

    print(f"\n🩺 Prediction Result: The patient is predicted to have **{prediction}**.")

if __name__ == "__main__":
    main()
