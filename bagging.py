#bagging.py
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

class BaggingModel:
    def __init__(self):
        self.model = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=10,
            random_state=42
        )

    def load_and_split_data(self, filepath):
        df = pd.read_csv(filepath)

        # Remove columns not needed
        df = df.drop(['affected', 'age_avg'], axis=1, errors='ignore')

        # Drop any rows with missing target value
        df = df.dropna(subset=['class'])

        # Convert target class to binary
        df['class'] = df['class'].str.strip().str.upper().map({'CKD': 1, 'NOT_CKD': 0})
        
        # Drop rows where mapping failed
        df = df[df['class'].notna()]
        df['class'] = df['class'].astype(int)

        # Optional: clean 'grf' too
        if 'grf' in df.columns:
            df['grf'] = df['grf'].str.strip().str.capitalize().map({'Normal': 0, 'Abnormal': 1})

        X = df.drop('class', axis=1)
        y = df['class']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        print("✅ Accuracy:", accuracy_score(self.y_test, y_pred))
        print("📊 Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("📋 Classification Report:\n", classification_report(self.y_test, y_pred))

if __name__ == "__main__":
    bag = BaggingModel()
    bag.load_and_split_data("ckd_prediction_dataset.csv")
    bag.train()
    bag.evaluate()