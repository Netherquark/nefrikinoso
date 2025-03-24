import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class SVMModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        self.model = None

    def preprocess_data(self):
        self.df = self.df.drop(['affected', 'age_avg'], axis=1)

        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])
        self.df['grf'] = le.fit_transform(self.df['grf'])

        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train_svm(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        classification = classification_report(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        print(f"\n--- Model Evaluation Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:\n", confusion)
        print("\nClassification Report:\n", classification)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        self.visualize_results(y_proba, confusion)

    def visualize_results(self, y_proba, confusion):
        print("\n--- Visualization of Results ---")

        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.histplot(y_proba, bins=10, kde=True, color='skyblue')
        plt.title("Histogram of Predicted Probabilities")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=self.y_test, y=y_proba)
        plt.title("Boxplot of Predicted Probabilities by True Class")
        plt.xlabel("True Class")
        plt.ylabel("Predicted Probability")
        plt.show()


if __name__ == "__main__":
    model = SVMModel('/Users/janhvidoijad/Desktop/Nefrikinoso/nefrikinoso/processed_data.csv')
    model.preprocess_data()

    print("Training SVM Model...")
    model.train_test_split()
    model.train_svm()
    model.evaluate_model()
