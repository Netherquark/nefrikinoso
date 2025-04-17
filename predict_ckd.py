import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class CKDPredictor:
    def __init__(self, model, encoders, scaler, feature_order):
        self.model = model
        self.feature_order = feature_order
        self.encoders = encoders  # Label encoders for categorical columns
        self.scaler = scaler  # Scaler for numerical features

    def preprocess_input(self, user_input):
        input_df = pd.DataFrame([user_input])

        # Handle categorical columns
        for col in self.encoders:
            input_df[col] = input_df[col].astype(str).str.lower().str.strip()
            le = self.encoders[col]
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                print(f"Warning: Unknown label for column '{col}', using default.")
                input_df[col] = 0

        # Fill missing values
        input_df = input_df.fillna(0)

        # Ensure correct feature order
        input_df = input_df[self.feature_order]

        # Apply scaling if scaler is available
        if self.scaler:
            input_scaled = self.scaler.transform(input_df)
        else:
            input_scaled = input_df.values  # No scaling

        return input_scaled

    def predict(self, user_input):
        processed_input = self.preprocess_input(user_input)
        prediction = self.model.predict(processed_input)[0]
        return "CKD" if prediction == 0 else "Not CKD"
