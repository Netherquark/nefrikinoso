from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class VotingEnsembleModel:
   def __init__(self, file_path):
       # Initialize the necessary attributes
       self.file_path = file_path
       self.df = pd.read_csv(file_path)  # Read the dataset
       self.model = None  # Placeholder for the final model


       # Attributes for processed data
       self.X = None  # Features
       self.y = None  # Target variable
       self.X_train = None  # Training features
       self.X_test = None  # Test features
       self.y_train = None  # Training target
       self.y_test = None  # Test target
       self.X_scaled = None  # Scaled features


       # Best models after tuning
       self.best_rf = None
       self.best_svm = None
       self.best_xgb = None
       self.best_lr = None


   def preprocess_data(self):
       # Drop unnecessary columns
       columns_to_drop = [col for col in ['affected', 'age_avg'] if col in self.df.columns]
       self.df.drop(columns=columns_to_drop, axis=1, inplace=True)


       # Label encoding for categorical columns
       le = LabelEncoder()
       if self.df['class'].dtype == 'object':
           self.df['class'] = le.fit_transform(self.df['class'])
       if self.df['grf'].dtype == 'object':
           self.df['grf'] = le.fit_transform(self.df['grf'])


       # Set features and target
       self.X = self.df.drop('class', axis=1)
       self.y = self.df['class']


   def train_test_split(self, test_size=0.2, random_state=42):
       # Standardize the features
       scaler = StandardScaler()
       self.X_scaled = scaler.fit_transform(self.X)


       # Split data into train and test sets
       self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
           self.X_scaled, self.y, test_size=test_size, random_state=random_state
       )


   def tune_models(self):
       # Random Forest tuning
       rf = RandomForestClassifier(random_state=42)
       rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
       rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
       rf_grid.fit(self.X_train, self.y_train)
       self.best_rf = rf_grid.best_estimator_


       # SVM tuning
       svm = SVC(probability=True, random_state=42)
       svm_params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
       svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='accuracy', n_jobs=-1)
       svm_grid.fit(self.X_train, self.y_train)
       self.best_svm = svm_grid.best_estimator_


       # XGBoost tuning
       xgboost = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
       xgb_params = {'n_estimators': [100, 150], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
       xgb_grid = GridSearchCV(xgboost, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
       xgb_grid.fit(self.X_train, self.y_train)
       self.best_xgb = xgb_grid.best_estimator_


       # Logistic Regression tuning
       logreg = LogisticRegression(max_iter=1000, random_state=42)
       logreg_params = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
       log_grid = GridSearchCV(logreg, logreg_params, cv=3, scoring='accuracy', n_jobs=-1)
       log_grid.fit(self.X_train, self.y_train)
       self.best_lr = log_grid.best_estimator_


   def train_model(self):
       # Make sure tuning is done before training
       self.tune_models()


       # Create the Voting Classifier
       self.model = VotingClassifier(
           estimators=[
               ('rf', self.best_rf),
               ('svm', self.best_svm),
               ('xgb', self.best_xgb),
               ('lr', self.best_lr)
           ],
           voting='soft'
       )
       # Train the model
       self.model.fit(self.X_train, self.y_train)


   def evaluate_model(self, return_scores=False):
       # Make predictions
       y_pred = self.model.predict(self.X_test)
       y_proba = self.model.predict_proba(self.X_test)[:, 1]


       # Calculate accuracy and ROC AUC
       accuracy = accuracy_score(self.y_test, y_pred)
       roc_auc = roc_auc_score(self.y_test, y_proba)


       # Return scores if needed
       if return_scores:
           return {"Accuracy": accuracy, "ROC_AUC": roc_auc}


       # Display evaluation results
       print(f"\nModel Evaluation for Voting Ensemble (Tuned):")
       print(f"Accuracy: {accuracy:.4f}")
       print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
       print("Classification Report:\n", classification_report(self.y_test, y_pred))
       print(f"ROC AUC Score: {roc_auc:.4f}")


       # Visualize results
       self.visualize_results(y_pred)


   def visualize_results(self, y_pred):
       # Plot the confusion matrix
       plt.figure(figsize=(6, 4))
       sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Greens')
       plt.title("Confusion Matrix - Tuned Voting Classifier")
       plt.xlabel("Predicted")
       plt.ylabel("Actual")
       plt.tight_layout()
       plt.show()


