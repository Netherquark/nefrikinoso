import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CKDPredictor:
    def __init__(self, model, feature_order):
        self.model = model
        self.feature_order = feature_order

        self.all_possible_categoricals = {
            'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
            'appet', 'pe', 'ane', 'grf', 'sex', 'hypertension'
        }
        self.categorical_cols = [col for col in self.feature_order if col in self.all_possible_categoricals]

    def get_user_input(self):
        print("\nEnter patient data for CKD prediction:")
        user_data = {}

        # Default schema assuming float if not in this mapping
        custom_types = {
            'al': int, 'su': int, 'pcv': int,
            'htn': str, 'dm': str, 'cad': str, 'appet': str,
            'pe': str, 'ane': str, 'grf': str, 'rbc': str,
            'pc': str, 'pcc': str, 'ba': str, 'sex': str,
            'hypertension': str
        }

        for feature in self.feature_order:
            dtype = custom_types.get(feature, float)
            value = input(f"{feature}: ")
            try:
                user_data[feature] = dtype(value)
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a valid {dtype.__name__} value.")
                return self.get_user_input()

        return user_data

    def preprocess_input(self, user_input):
        df = pd.DataFrame([user_input], columns=self.feature_order)

        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        return df_scaled

    def predict(self, user_input):
        processed_input = self.preprocess_input(user_input)
        pred = self.model.predict(processed_input)[0]
        return "CKD" if pred == 0 else "NOT_CKD"
