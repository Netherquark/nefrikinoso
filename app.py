from flask import Flask, render_template, request
from main import CKDModelRunner

app = Flask(__name__)
model = CKDModelRunner(dataset_path="ckd_prediction_dataset.csv")  # Make sure this has `predict_from_user_input_gui` method

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Create a dictionary for user input
        user_input_dict = {
            "age": float(request.form.get('age', 0)),
            "bp": float(request.form.get('bp', 0)),
            "sg": float(request.form.get('sg', 0)),
            "al": float(request.form.get('al', 0)),
            "su": float(request.form.get('su', 0)),
            "rbc": request.form.get('rbc', ''),
            "pc": request.form.get('pc', ''),
            "pcc": request.form.get('pcc', ''),
            "ba": request.form.get('ba', ''),
            "bgr": float(request.form.get('bgr', 0)),
            "bu": float(request.form.get('bu', 0)),
            "sc": float(request.form.get('sc', 0)),
            "sod": float(request.form.get('sod', 0)),
            "pot": float(request.form.get('pot', 0)),
            "hemo": float(request.form.get('hemo', 0)),
            "pcv": float(request.form.get('pcv', 0)),
            "wbcc": float(request.form.get('wbcc', 0)),
            "rbcc": float(request.form.get('rbcc', 0)),
            "htn": request.form.get('htn', ''),
            "dm": request.form.get('dm', ''),
        }

        prediction = model.predict_from_user_input_gui(user_input_dict)
        result = "CKD Detected" if prediction == "CKD" else "No CKD Detected"

        return render_template('index.html', user_input=user_input_dict, predictions={'Stacked Ensemble Learning': result}, final_conclusion=prediction)

    # Default CKD-indicative values to prefill form
    default_input = {
        "age": 65, "bp": 80, "sg": 1.005, "al": 3, "su": 2,
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "present",
        "bgr": 150, "bu": 60, "sc": 3.5, "sod": 140, "pot": 5.0,
        "hemo": 9.5, "pcv": 30, "wbcc": 12000, "rbcc": 3.5,
        "htn": "yes", "dm": "yes"
    }

    return render_template('index.html', user_input=default_input)


if __name__ == '__main__':
    app.run(debug=True, port=5000)