import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

class DecisionTreeCKD:
    def __init__(self, file_path, target_column, n_splits=5):
        self.file_path = file_path
        self.target_column = target_column
        self.n_splits = n_splits
        self.dataset = None
        self.results = []
        self.best_params = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.X = None
        self.y = None

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
        self.X = self.dataset.drop(columns=[self.target_column])
        self.y = self.dataset[self.target_column]

        self.tune_hyperparameters(self.X, self.y)

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            self.model = DecisionTreeClassifier(**self.best_params, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Save last test split for visualization
            self.X_test, self.y_test = X_test, y_test

            self.results.append(self.evaluate_metrics(y_test, y_pred))

        print("[INFO] Cross-validation complete.")

    def get_results(self):
        results_array = np.array(self.results)
        accuracy, sensitivity, specificity, precision, recall, f1 = results_array.mean(axis=0)
        return (f"[Decision Tree] Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, "
                f"Specificity: {specificity:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    def visualize_results(self):
        if self.model is None or self.X_test is None or self.y_test is None:
            print("[ERROR] No model or test data available for visualization.")
            return

        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Accuracy & Report
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.plot(fpr, tpr, label="ROC Curve", color="darkorange")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

        # Feature Importance
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            features = self.X.columns
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
            plt.title("Feature Importance")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.show()

        # Decision Tree Visualization
        plt.figure(figsize=(15, 10))
        plot_tree(self.model, feature_names=self.X.columns, class_names=["Not CKD", "CKD"], filled=True)
        plt.title("Decision Tree Visualization")
        plt.show()

        print("[INFO] Visualizations generated.")

# ---------------------- Usage ---------------------- #
file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
target_column = "class"

dt_ckd = DecisionTreeCKD(file_path, target_column)
dt_ckd.preprocess_data()
dt_ckd.cross_validate()
print(dt_ckd.get_results())
dt_ckd.visualize_results()
