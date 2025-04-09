import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


class CKDEnsembleModel:
    def __init__(self):
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.voting_model = None
        self.stacking_model = None

    def preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        df = df.drop(['affected', 'age_avg'], axis=1, errors='ignore')

        # Encode categorical columns
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])  # CKD=0, NOT_CKD=1
        df['grf'] = le.fit_transform(df['grf'])

        self.X = df.drop('class', axis=1)
        self.y = df['class']
        self.feature_names = self.X.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def build_models(self):
        # Define base models
        lr = LogisticRegression(max_iter=1000)
        svm = SVC(kernel='rbf', probability=True)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        # Voting Classifier
        self.voting_model = VotingClassifier(
            estimators=[('lr', lr), ('svm', svm), ('rf', rf), ('xgb', xgb)],
            voting='soft'
        )

        # Stacking Classifier
        self.stacking_model = StackingClassifier(
            estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
            final_estimator=xgb,
            passthrough=True
        )

    def train_and_evaluate(self, model, model_name):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred, output_dict=True)

        print(f"\n🔍 Evaluation for {model_name} Ensemble")
        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Confusion Matrix:")
        print(cm)
        print("📋 Classification Report:")
        print(classification_report(self.y_test, y_pred))

        self.visualize_metrics(cm, cr, model_name)

    def visualize_metrics(self, cm, report_dict, model_name):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{model_name} Ensemble Metrics", fontsize=16)

        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0])
        axs[0].set_title("Confusion Matrix")
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("Actual")

        # Classification report table
        report_df = pd.DataFrame(report_dict).transpose()
        axs[1].axis('off')
        table = axs[1].table(
            cellText=np.round(report_df.values, 2),
            colLabels=report_df.columns,
            rowLabels=report_df.index,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.4, 1.4)
        axs[1].set_title("Classification Report")

        plt.tight_layout()
        plt.show()

    def run_all(self, file_path):
        self.preprocess_data(file_path)
        self.build_models()

        self.train_and_evaluate(self.voting_model, "Voting")
        self.train_and_evaluate(self.stacking_model, "Stacking")


# ----------------- MAIN -----------------

if __name__ == "__main__":
    model = CKDEnsembleModel()
    model.run_all("ckd_prediction_dataset.csv")
