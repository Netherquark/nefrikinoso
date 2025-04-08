import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler

class CKD_KNN_CrossValidator:
    def __init__(self, file_path, target_column="class", n_splits=5):
        self.file_path = file_path
        self.dataset = None
        self.target_column = target_column
        self.n_splits = n_splits
        self.best_k = None
        self.statistics = []
        self.best_fold_data = {}  # Store data for best fold (for plotting)
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        self.dataset = pd.read_csv(self.file_path)
        self.dataset.drop(columns=["affected", "grf"], inplace=True)
        self.dataset[self.target_column] = self.dataset[self.target_column].map({'ckd': 1, 'notckd': 0})
        self.dataset = self.dataset.apply(pd.to_numeric, errors='coerce')
        self.dataset.fillna(self.dataset.median(), inplace=True)
        scaler = StandardScaler()
        features = self.dataset.drop(columns=[self.target_column])
        self.dataset[features.columns] = scaler.fit_transform(features)

    def tune_hyperparameters(self):
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        param_grid = {'n_neighbors': range(1, 20, 2)}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)
        self.best_k = grid_search.best_params_['n_neighbors']
        print(f"[INFO] Best k: {self.best_k}")

    def evaluate_model(self):
        if self.best_k is None:
            self.tune_hyperparameters()
        
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        best_acc = 0
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = KNeighborsClassifier(n_neighbors=self.best_k)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            cm = confusion_matrix(y_test, y_pred)
            accuracy = np.trace(cm) / np.sum(cm)
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
            self.statistics.append([accuracy, sensitivity, specificity])

            # Save best fold data
            if accuracy > best_acc:
                best_acc = accuracy
                self.best_fold_data = {
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'model': model,
                    'X_test': X_test
                }

    def get_results(self):
        stats_df = pd.DataFrame(self.statistics, columns=["Accuracy", "Sensitivity", "Specificity"])
        mean_accuracy = stats_df["Accuracy"].mean()
        mean_sensitivity = stats_df["Sensitivity"].mean()
        mean_specificity = stats_df["Specificity"].mean()
        print(f"[K-NN Results] Accuracy: {mean_accuracy:.4f}, Sensitivity: {mean_sensitivity:.4f}, Specificity: {mean_specificity:.4f}")
        return mean_accuracy, mean_sensitivity, mean_specificity

    def plot_results(self):
        if not self.best_fold_data:
            print("No evaluation data found. Run evaluate_model() first.")
            return

        y_test = self.best_fold_data['y_test']
        y_pred = self.best_fold_data['y_pred']
        y_proba = self.best_fold_data['y_proba']

        print("\n[Classification Report]\n", classification_report(y_test, y_pred))
        print("[Accuracy]:", accuracy_score(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label="ROC Curve", color="darkorange")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

        print("\nNote: Feature importance is not available for K-NN.")

# === USAGE ===
file_path = "/home/r1ddh1/2nd_year/pbl_sem4/processed_data.csv"
ckd_knn_validator = CKD_KNN_CrossValidator(file_path)
ckd_knn_validator.evaluate_model()
ckd_knn_validator.get_results()
ckd_knn_validator.plot_results()
