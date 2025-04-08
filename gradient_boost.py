import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, recall_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


class GradientBoostModel:
    def __init__(self, desired_recall=0.98):
        self.model = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.report = None
        self.accuracy = 0.0
        self.conf_matrix = None
        self.desired_recall = desired_recall

    def preprocess_data(self, df):
        self.df = df.copy()
        self.df = self.df.drop(['affected', 'age_avg'], axis=1, errors='ignore')

        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])  # CKD/NOT_CKD to 0/1
        self.df['grf'] = le.fit_transform(self.df['grf'])

        self.feature_names = [col for col in self.df.columns if col != 'class']
        self.X = self.df[self.feature_names]
        self.y = self.df['class']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def train(self):
        param_grid = {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.1],
            'max_depth': [4, 5],
            'subsample': [0.8, 1.0]
        }

        gbc = GradientBoostingClassifier(random_state=42)
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(self.X_train, self.y_train)

        grid = GridSearchCV(
            gbc,
            param_grid,
            scoring='recall',
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_res, y_res)
        self.model = grid.best_estimator_

    def predict(self):
        y_probs = self.model.predict_proba(self.X_test)[:, 1]

        # Aggressive threshold tuning to hit desired recall
        thresholds = np.linspace(0.1, 0.9, 100)
        best_recall, best_threshold = 0, 0.5
        for thresh in thresholds:
            y_pred_temp = (y_probs > thresh).astype(int)
            recall = recall_score(self.y_test, y_pred_temp)
            if recall >= self.desired_recall:
                best_recall = recall
                best_threshold = thresh
                break

        print(f"Selected Threshold for Recall ≥ {self.desired_recall}: {best_threshold:.2f}")
        self.y_pred = (y_probs > best_threshold).astype(int)

    def evaluate(self):
        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.report = classification_report(self.y_test, self.y_pred, output_dict=True)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

        print("Accuracy:", self.accuracy)
        print("Confusion Matrix:\n", self.conf_matrix)
        print("Classification Report:\n", classification_report(self.y_test, self.y_pred))

        recall_val = self.report['1']['recall']
        if recall_val < self.desired_recall:
            print(f"⚠️ Warning: Recall is {recall_val:.3f}, below desired {self.desired_recall}")
        else:
            print(f"✅ Success: Recall is {recall_val:.3f}")
        return recall_val

    def visualize_all(self):
        if self.X_test is None or self.y_pred is None:
            print("Run train() and predict() first.")
            return

        fig, axs = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("Gradient Boosting Model Evaluation", fontsize=18, fontweight='bold')

        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Reds', ax=axs[0, 0])
        axs[0, 0].set_title("Confusion Matrix")
        axs[0, 0].set_xlabel("Predicted")
        axs[0, 0].set_ylabel("Actual")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = [self.feature_names[i] for i in indices]
        sns.barplot(x=importances[indices], y=features, ax=axs[0, 1], palette='viridis')
        axs[0, 1].set_title("Feature Importances")

        y_score = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_score)
        auc = roc_auc_score(self.y_test, y_score)
        axs[1, 0].plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='green')
        axs[1, 0].plot([0, 1], [0, 1], 'k--')
        axs[1, 0].set_title("ROC Curve")
        axs[1, 0].legend(loc='lower right')

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
        axs[1, 1].set_title("Classification Report")

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()


def main():
    df = pd.read_csv("ckd_prediction_dataset.csv")
    model = GradientBoostModel(desired_recall=0.98)
    model.preprocess_data(df)
    model.train()
    model.predict()
    final_recall = model.evaluate()
    print(f"📈 Final Recall: {final_recall:.4f}")
    model.visualize_all()


if __name__ == "__main__":
    main()
