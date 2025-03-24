import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.model = None

    def preprocess_data(self):
        # Remove redundant columns
        self.df = self.df.drop(['affected', 'age_avg'], axis=1)

        # Encode categorical columns
        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])
        self.df['grf'] = le.fit_transform(self.df['grf'])

        # Define features and target
        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
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

    def plot_feature_importance(self):
        importance = self.model.feature_importances_
        features = self.X.columns

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features)
        plt.title('Feature Importance')
        plt.show()


if __name__ == "__main__":
    model = XGBoostModel('/Users/janhvidoijad/Desktop/Nefrikinoso/nefrikinoso/processed_data.csv')
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.plot_feature_importance()
