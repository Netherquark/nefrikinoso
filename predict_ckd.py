import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CKDPredictor:
    def __init__(self, model, feature_order):
        self.model = model
        self.feature_order = feature_order
        self.encoders = {}
        self.scaler = None
        self.cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm'] # Define categorical columns here

        self.setup_encoders_and_scaler()

    def setup_encoders_and_scaler(self, encoder_path="encoders.pkl", scaler_path="scaler.pkl"):
        try:
            with open(encoder_path, 'rb') as file:
                self.encoders = pickle.load(file)
            logging.info("Encoders loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading encoders: {e}")

        try:
            with open(scaler_path, 'rb') as file:
                self.scaler = pickle.load(file)
            logging.info("Scaler loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading scaler: {e}")

    def preprocess_input(self, user_input):
        input_df = pd.DataFrame([user_input])

        # Encode categorical features
        for col in self.cat_cols:
            if col in self.encoders:
                le = self.encoders[col]
                v = str(input_df.at[0, col]).lower().strip()
                if pd.isna(v) or v == 'nan':
                    v = 'missing' # Or your default strategy
                if v not in le.classes_:
                    logging.warning(f"Unseen value '{v}' for column '{col}'. Using a default strategy (first class).")
                    v = le.classes_[0]
                input_df.at[0, col] = le.transform([v])[0]
            else:
                logging.warning(f"Encoder not found for column: {col}")
                input_df[col] = input_df[col].astype('category').cat.codes # Fallback

        # Fill any missing values in numerical columns (if any after user input)
        numerical_cols = [col for col in self.feature_order if col not in self.cat_cols]
        input_df[numerical_cols] = input_df[numerical_cols].apply(pd.to_numeric, errors='coerce').fillna(input_df[numerical_cols].median())


        # Ensure correct feature order
        input_df = input_df[self.feature_order]

        return input_df

    def scale_input(self, input_df):
        # Only scale numerical columns
        numerical_cols = [col for col in self.feature_order if col not in self.cat_cols]
        if self.scaler is not None:
            input_df[numerical_cols] = self.scaler.transform(input_df[numerical_cols])
        else:
            logging.error("Scaler not loaded. Prediction might be inaccurate.")
        return input_df

    def predict_from_input_dict(self, user_input_dict):
        # Ensure feature order
        input_df = pd.DataFrame([user_input_dict])[self.feature_order]
        # Preprocess input
        processed_input = self.preprocess_input(input_df)
        processed_input = self.scale_input(processed_input)
        prediction = self.model.predict(processed_input)
        return "CKD" if prediction[0] == 1 else "Not CKD"