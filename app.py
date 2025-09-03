from flask import Flask, render_template, request
import os
import json
import pypdf
import google.generativeai as genai
import re
from main import CKDModelRunner

# --- Gemini API Configuration ---
# IMPORTANT: Add your Gemini API key here.
# It is highly recommended to use environment variables for security.
# For example: genai.configure(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_API_KEY = "ADD_YOUR_GEMINI_API_KEY_HERE"
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

app = Flask(__name__)

# --- Correct file paths ---
# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "ckd_prediction_dataset.csv")
model_path = os.path.join(script_dir, "best_model.pkl")
feature_order_path = os.path.join(script_dir, "feature_order.json")

model = CKDModelRunner(dataset_path=dataset_path, best_model_path=model_path)

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        reader = pypdf.PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def get_params_from_gemini(pdf_text, feature_list, max_retries=2):
    """
    Uses Gemini API to extract medical parameters from PDF text with a self-correcting retry loop and robust parsing.
    """
    if not pdf_text:
        return None

    gemini_model = genai.GenerativeModel('gemini-pro')
    
    base_prompt = f"""
    You are a world-class medical data extraction AI. Your only task is to analyze medical text and return a valid JSON object.

    **Instructions:**
    1.  **Output MUST be JSON only.** Do not include any other text, explanations, or markdown like ```json.
    2.  **Use Exact Keys:** The JSON keys must exactly match the "Key" from the Parameter Table.
    3.  **Extract Values:**
        - For numbers (e.g., "bp", "age"), extract only the numerical value. Use the systolic value for Blood Pressure.
        - For categories (e.g., "rbc", "htn"), map the text to one of the allowed values: 'normal'/'abnormal', 'present'/'notpresent', 'yes'/'no'.
    4.  **Handle Missing Data:** If a value is not found in the text, use `0` for numbers and an empty string `""` for categories. Every key must be present in the final JSON.

    **Parameter Table:**
    - `age`: Age (years)
    - `bp`: Blood Pressure (systolic, mmHg)
    - `sg`: Specific Gravity
    - `al`: Albumin
    - `su`: Sugar
    - `rbc`: Red Blood Cells
    - `pc`: Pus Cell
    - `pcc`: Pus Cell Clumps
    - `ba`: Bacteria
    - `bgr`: Blood Glucose Random
    - `bu`: Blood Urea
    - `sc`: Serum Creatinine
    - `sod`: Sodium
    - `pot`: Potassium
    - `hemo`: Hemoglobin
    - `pcv`: Packed Cell Volume
    - `wbcc`: White Blood Cell Count
    - `rbcc`: Red Blood Cell Count
    - `htn`: Hypertension
    - `dm`: Diabetes Mellitus

    **Medical Report Text:**
    ---
    {pdf_text}
    ---
    """

    for attempt in range(max_retries + 1):
        try:
            prompt = base_prompt
            if attempt > 0:
                # Add a self-correction instruction on retries
                prompt += "\nYour previous attempt failed. Please re-analyze the text and ensure the output is ONLY a single, valid JSON object. Do not include any extra text or formatting."

            response = gemini_model.generate_content(prompt)
            
            # Advanced JSON parsing
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            
            if json_match:
                json_text = json_match.group(0)
                # Final validation
                parsed_json = json.loads(json_text)
                # Ensure all keys are present, if not, it will fail and retry
                if all(key in parsed_json for key in feature_list):
                    return parsed_json # Success!

            print(f"Attempt {attempt + 1}: Failed to get a valid, complete JSON response. Retrying...")
            print("--- Gemini Raw Response ---")
            print(response.text)
            print("---------------------------")

        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            if 'response' in locals():
                print("--- Gemini Raw Response on Error ---")
                print(response.text)
                print("------------------------------------")
    
    print("All retry attempts failed.")
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    error_message = None

    if request.method == 'POST':
        pdf_files = request.files.getlist('pdf_files') # Use getlist for multiple files

        if not pdf_files or pdf_files[0].filename == '':
            error_message = "No PDF files were uploaded. Please select one or more files."
            return render_template('index.html', error=error_message)

        # Load feature order from the JSON file
        try:
            with open(feature_order_path, "r") as f:
                feature_keys = json.load(f)
        except FileNotFoundError:
            error_message = "Critical error: The 'feature_order.json' file is missing."
            return render_template('index.html', error=error_message)

        for pdf_file in pdf_files:
            pdf_text = extract_text_from_pdf(pdf_file)
            
            if pdf_text:
                user_input_dict = get_params_from_gemini(pdf_text, feature_keys)
                
                if user_input_dict:
                    prediction = model.predict_from_user_input_gui(user_input_dict)
                    result = "CKD Detected" if prediction == "CKD" else "No CKD Detected"
                    predictions.append({
                        'filename': pdf_file.filename,
                        'data': user_input_dict,
                        'prediction': result,
                        'conclusion': prediction
                    })
                else:
                    predictions.append({
                        'filename': pdf_file.filename,
                        'error': "Failed to extract data from this PDF using the AI model."
                    })
            else:
                predictions.append({
                    'filename': pdf_file.filename,
                    'error': "Could not read text from this PDF."
                })

    return render_template('index.html', predictions=predictions, error=error_message)


if __name__ == '__main__':
    app.run(debug=True, port=5000)