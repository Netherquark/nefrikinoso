import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class XGBoostModel:
    def __init__(self, file_path):
        # Initialize with file path and read the dataset
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)  # Ensure the file path is correct
        self.model = None  # Placeholder for the trained model

        # Initialize the data attributes
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_scaled = None

    def preprocess_data(self):
        # Drop unused columns
        drop_cols = ['index','affected', 'age_avg', 'stage']
        self.df.drop(columns=[col for col in drop_cols if col in self.df.columns], inplace=True)

        # Label encode target and 'grf' if needed
        le = LabelEncoder()
        if self.df['class'].dtype == 'object':
            self.df['class'] = le.fit_transform(self.df['class'])
        if 'grf' in self.df.columns and self.df['grf'].dtype == 'object':
            self.df['grf'] = le.fit_transform(self.df['grf'])

        # Set features and target
        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

        # Categorical column handling
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
                            'appet', 'pe', 'ane', 'grf', 'sex', 'hypertension']
        categorical_cols = [col for col in categorical_cols if col in self.X.columns]

        for col in categorical_cols:
            self.X[col] = self.X[col].astype(str).str.lower().str.strip()
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])

        # Fill missing values
        self.X = self.X.fillna(0)

        # Scale features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def train_test_split(self, test_size=0.2, random_state=42):
        # Split data into train and test sets after preprocessing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        # Set up parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 4],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }

        # Initialize the XGBoost classifier
        xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        # Fit the model
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

        # Print the best parameters found during grid search
        print("\nBest Parameters Found for XGBoost:")
        print(grid_search.best_params_)

    def evaluate_model(self, return_scores=False):
        # Make predictions and compute probabilities
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Calculate accuracy and ROC AUC
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        # Return the scores if required
        if return_scores:
            return {
                "Accuracy": accuracy,
                "ROC_AUC": roc_auc
            }

        # Print the evaluation results
        print(f"\nModel Evaluation for XGBoost:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.4f}")