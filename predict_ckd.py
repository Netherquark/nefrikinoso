import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CKDPredictor:
    def __init__(self, model, feature_order):
        self.model = model
        self.feature_order = feature_order
        self.encoders = {}  # Store label encoders for categorical columns
        self.scaler = StandardScaler()  # Initialize the scaler
        self.cat_cols = []  # To track categorical columns
        
        # Preprocess feature order and setup
        self.setup_encoders_and_scaler()
        
    def get_user_input(self):
        print("\nPlease enter the following details for CKD prediction:")
        user_input = {}
        for feature in self.feature_order:
            while True:
                try:
                    if feature in self.encoders:  # Categorical feature
                        options = list(self.encoders[feature].classes_)
                        options_str = "/".join(options)
                        val = input(f"Enter {feature} ({options_str}): ").strip().lower()
                        if val not in options:
                            raise ValueError(f"Invalid option. Choose from: {options_str}")
                        user_input[feature] = val
                    else:  # Numerical feature
                        val = input(f"Enter {feature} (numerical): ").strip()
                        user_input[feature] = float(val)
                    break
                except ValueError as ve:
                    print(f"Invalid input: {ve}")
                except Exception:
                    print("Invalid input. Please try again.")
        return user_input

    def setup_encoders_and_scaler(self):
        try:
            dataset = pd.read_csv("ckd_prediction_dataset.csv")

            # Define known categorical columns (ensure this matches your dataset)
            self.cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm']
            self.cat_cols = [col for col in self.cat_cols if col in dataset.columns]
            num_cols = [col for col in self.feature_order if col not in self.cat_cols]

            # Encode categorical columns
            for col in self.cat_cols:
                le = LabelEncoder()
                dataset[col] = dataset[col].astype(str).str.lower().str.strip()
                le.fit(dataset[col])
                self.encoders[col] = le

            # Create encoded dataset for scaling
            dataset_enc = dataset.copy()
            for col, le in self.encoders.items():
                dataset_enc[col] = le.transform(dataset[col])
            dataset_enc = dataset_enc[num_cols]  # Only numerical cols
            self.scaler.fit(dataset_enc)
        except Exception as e:
            print(f"Error during setup: {e}")

    def preprocess_input(self, user_input):
        input_df = pd.DataFrame([user_input])[self.feature_order]

        # Encode categorical features
        for col in self.encoders:
            input_df[col] = input_df[col].astype(str).str.lower().str.strip()
            le = self.encoders[col]
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                print(f"Warning: Unknown label for column '{col}', using default.")
                input_df[col] = 0

        # Fill any missing values
        input_df = input_df.fillna(0)

        # Ensure correct feature order
        input_df = input_df[self.feature_order]

        return input_df

    def scale_input(self, input_df):
        # Only scale numerical columns
        num_cols = [col for col in self.feature_order if col not in self.encoders]
        input_df[num_cols] = self.scaler.transform(input_df[num_cols])
        return input_df

    def predict(self, user_input):
        processed_input = self.preprocess_input(user_input)

        processed_input = self.scale_input(processed_input)

        prediction = self.model.predict(processed_input)[0]
        return "CKD" if prediction == 0 else "Not CKD"
    
    def predict_from_input_dict(self, user_input_dict):
        # Preprocess input
        processed_input = self.preprocess_input(user_input_dict)
        processed_input = self.scale_input(processed_input)
        prediction = self.model.predict(processed_input)
        return "CKD" if prediction[0] == 0 else "Not CKD"