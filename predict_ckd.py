import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CKDPredictor:
    def __init__(self, model, feature_order, le_dict=None, scaler=None):
        """
        Initializes the CKD predictor with a trained model and the feature order.
        
        :param model: Trained machine learning model for CKD prediction (e.g., XGBoost)
        :param feature_order: List of features in the correct order as expected by the model
        :param le_dict: Optional dictionary of LabelEncoders for categorical columns
        :param scaler: Optional scaler for standardizing input features
        """
        self.model = model
        self.feature_order = feature_order

        # List of possible categorical columns
        self.all_possible_categoricals = {
            'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
            'appet', 'pe', 'ane', 'grf', 'sex', 'hypertension'
        }
        
        self.categorical_cols = [col for col in self.feature_order if col in self.all_possible_categoricals]
        
        # Optional: Preload the LabelEncoders and StandardScaler if provided
        self.le_dict = le_dict or {}
        self.scaler = scaler or StandardScaler()

    def get_user_input(self):
        """
        Prompts the user for input to enter the CKD features.
        :return: Dictionary of user input values.
        """
        print("\nEnter patient data for CKD prediction:")
        user_data = {}

        # Define custom types for each feature (if necessary)
        custom_types = {
            'al': int, 'su': int, 'pcv': int,
            'htn': str, 'dm': str, 'cad': str, 'appet': str,
            'pe': str, 'ane': str, 'grf': str, 'rbc': str,
            'pc': str, 'pcc': str, 'ba': str, 'sex': str,
            'hypertension': str
        }

        for feature in self.feature_order:
            dtype = custom_types.get(feature, float)  # Default to float if not listed
            value = input(f"{feature}: ")
            try:
                user_data[feature] = dtype(value)
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a valid {dtype.__name__} value.")
                return self.get_user_input()  # Recurse if there's an invalid input

        return user_data

    def preprocess_input(self, user_input):
        """
        Preprocess the user input data (encoding and scaling).
        :param user_input: User input dictionary
        :return: Scaled numpy array for model prediction
        """
        df = pd.DataFrame([user_input], columns=self.feature_order)

        # Encode categorical columns using LabelEncoder
        for col in self.categorical_cols:
            # Use pre-fitted LabelEncoders for consistency in transformation
            if col not in self.le_dict:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.le_dict[col] = le
            else:
                df[col] = self.le_dict[col].transform(df[col])

        # Scale the features using the pre-fitted StandardScaler
        df_scaled = self.scaler.transform(df)  # Transform based on the pre-fitted scaler
        return df_scaled

    def predict(self, user_input):
        """
        Predict CKD status based on user input.
        :param user_input: User input dictionary for prediction
        :return: Predicted CKD status ("CKD" or "NOT_CKD")
        """
        processed_input = self.preprocess_input(user_input)
        pred = self.model.predict(processed_input)[0]  # Get the prediction result
        return "CKD" if pred == 0 else "NOT_CKD"
