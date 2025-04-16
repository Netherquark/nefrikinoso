
# nefrikinoso: Early Chronic Kidney Disease Prediction System
<img src="https://github.com/user-attachments/assets/042419b4-11c2-4e9c-a22d-1ab968007674" alt="nefrikinoso" width="170" />

## Overview
nefrikinoso is a machine learning project focused on predicting Chronic Kidney Disease (CKD). It utilizes various machine learning models to provide accurate CKD predictions and includes tools for model evaluation and user interaction.

## Features
* Preprocesses and prepares the CKD dataset.
* Trains and evaluates model performance using relevant metrics.
* Offers a web interface for user interaction and predictions.
* Provides an API endpoint for making predictions programmatically.
* Generates visualizations for model evaluation and feature analysis.
- Includes the following implementations:
  - **Novel 🤖** Voting Ensemble
  - **Novel 🤖** Stacked Ensemble Learning
  - XGBoost
  - SVM
  - Decision Tree
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Random Forest
  - Gradient Boosting
  - CatBoost
  - Neural Network

## Usage (Docker)
`docker run netherquark/nefrikinoso`

## Installation (Regular)
1. Clone the repository.
2. Navigate to the project directory.
3. Install the required dependencies using `pip install -r requirements.txt`.

## Usage (Regular)
### Training and Visualisation
Run the models and generate visualisations using `python main.py`.
### Running the Web Application
Execute the `app.py` script to launch the web interface for CKD prediction.

### Using the API
The API endpoint `/api/predict` accepts patient data for CKD prediction.

### Evaluating Models
Run the `main.py` script to evaluate and compare the performance of the implemented machine learning models.

## Dependencies
* Python 3.11
* pandas
* matplotlib
* joblib
* seaborn
* scikit-learn
* CatBoost
* XGBoost
* Flask
* GUnicorn

## License
This project is licensed under the GNU GPLv3 License. Refer to [LICENSE](https://github.com/Netherquark/nefrikinoso/blob/main/LICENSE) for more details.