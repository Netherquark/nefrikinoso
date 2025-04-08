import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder


class AdaBoostModel:
    def __init__(self, base_model=None):
        if base_model is None:
            base_model = DecisionTreeClassifier(max_depth=2, class_weight='balanced')  # Improved base model

        self.model = AdaBoostClassifier(
            estimator=base_model,
            n_estimators=100,              
            learning_rate=0.5,             
            random_state=42
        )


        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = None
        self.feature_names = None
        self.conf_matrix = None
        self.report = None
        self.accuracy = None
        self.recall = None
        self.df = None


    def preprocess_data(self, df):
        self.df = df.copy()

        # Drop unwanted columns
        self.df = self.df.drop(['affected', 'age_avg'], axis=1, errors='ignore')

        # Encode categorical columns
        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])  # Map CKD/NOT_CKD to 0/1
        self.df['grf'] = le.fit_transform(self.df['grf'])

        # Define feature matrix and target
        self.feature_names = [col for col in self.df.columns if col != 'class']
        self.X = self.df[self.feature_names]
        self.y = self.df['class']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_probs = self.model.predict_proba(self.X_test)[:, 1]
        self.y_pred = (y_probs > 0.3).astype(int)

    def evaluate(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred, average='macro')
        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.report = classification_report(self.y_test, self.y_pred, output_dict=True)

    def visualize_all(self):
        fig, axs = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("AdaBoost Model Evaluation", fontsize=18, fontweight='bold')

        # --- 1. Confusion Matrix ---
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Reds', ax=axs[0, 0])
        axs[0, 0].set_title("Confusion Matrix", fontsize=14)
        axs[0, 0].set_xlabel("Predicted Labels")
        axs[0, 0].set_ylabel("True Labels")

        # --- 2. Feature Importances ---
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = [self.feature_names[i] for i in indices]
        sns.barplot(x=importances[indices], y=features, ax=axs[0, 1], palette='viridis')
        axs[0, 1].set_title("Top Features", fontsize=14)
        axs[0, 1].set_xlabel("Importance Score")
        axs[0, 1].set_ylabel("Features")

        # --- 3. ROC Curve ---
        y_score = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_score)
        auc = roc_auc_score(self.y_test, y_score)
        axs[1, 0].plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='green')
        axs[1, 0].plot([0, 1], [0, 1], 'k--')
        axs[1, 0].set_title("ROC Curve", fontsize=14)
        axs[1, 0].set_xlabel("False Positive Rate")
        axs[1, 0].set_ylabel("True Positive Rate")
        axs[1, 0].legend(loc='lower right')

        # --- 4. Classification Report ---
        report_df = pd.DataFrame(self.report).transpose()
        axs[1, 1].axis('off')
        table = axs[1, 1].table(
            cellText=np.round(report_df.values, 2),
            colLabels=report_df.columns,
            rowLabels=report_df.index,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.4, 1.5)
        axs[1, 1].set_title("Classification Metrics", fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    def summary(self):
        print(" Accuracy:", round(self.accuracy, 4))
        print(" Recall:", round(self.recall, 4))

    def run_all(self, df, target_column="class"):
        self.preprocess_data(df)
        self.train()
        self.predict()
        self.evaluate()
        self.summary()
        self.visualize_all()


# ------------------ MAIN ------------------

if __name__ == "__main__":
    df = pd.read_csv("ckd_prediction_dataset.csv")  # Replace with your file path
    adb = AdaBoostModel()
    adb.run_all(df)
