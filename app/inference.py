# Libraries
import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Custom
from app.variables import MODEL_FEATURES
from app.pre_processing import Preprocessor
from app.data_validation import DataValidator

# Paths
MODEL_PATH = 'models/classifier.pkl'
SAMPLE_INPUT_DATA = 'data/dataset.csv'
INFERENCE_RESULTS_PATH = 'data/inference_results.csv'

# Initialize Flask app
appContainer = Flask(__name__)

# Sample Prediction
@appContainer.route("/predict", methods=["POST"])
def inference_sample_pipeline():
    '''
    Perform the inference process for a given sample
    Args:
        data_path (str): Path to the input CSV file containing the data for inference.
    Saves:
        CSV file with the predicted labels and probabilities appended to the input data.
    Returns:
        None
    '''
    # Check model and imputer existance
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model file not found at {MODEL_PATH}. Ensure the model is trained and saved.')
    
    # Check input from post request
    if not request.is_json:
        return jsonify({"error": "Invalid content type, expecting application/json"}), 400
    
    # Load imputer and model 
    mdl = pickle.load(open(MODEL_PATH, "rb"))

    # Load data to be infer
    data = pd.DataFrame(request.json)

    # Validate the data
    data = DataValidator().validate(data)

    # Specify the features (validated before)
    data = data[MODEL_FEATURES].copy()

    # Preprocess the data
    x = Preprocessor(is_training=False).apply_preprocessing(data)

    # Predict using the loaded model
    y_pred = mdl.predict(x).tolist()
    y_proba = mdl.predict_proba(x).tolist()

    # Return results
    return jsonify({"prediction": y_pred, 'prediction_proba': y_proba}), 200

@appContainer.route("/predict_batch", methods=["POST"])
def inference_batch_pipeline(data_path):
    '''
    Perform the inference process for a given dataset using a pre-trained model.
    Args:
        data_path (str): Path to the input CSV file containing the data for inference.
    Saves:
        CSV file with the predicted labels and probabilities appended to the input data.
    Returns:
        None
    '''
    # Check model and imputer existance
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model file not found at {MODEL_PATH}. Ensure the model is trained and saved.')
    
    # Load imputer and model 
    mdl = pickle.load(open(MODEL_PATH, "rb"))

    # Load data to be infer
    data = pd.read_csv(data_path)

    # Validate the data
    data = DataValidator().validate(data)

    # Specify the features (validated before)
    data = data[MODEL_FEATURES].copy()

    # Preprocess the data
    x = Preprocessor(is_training=False).apply_preprocessing(data)

    # Predict using the loaded model
    y_pred = mdl.predict(x)
    y_proba = mdl.predict_proba(x)

    # Save results predictions in the dataframe
    data['PREDICTED_LABEL'] = y_pred
    data['PROBABILITY_POS'] = y_proba[:, 1]

    # Save results
    print("Inference Finished! Results saved.")
    data.to_csv(INFERENCE_RESULTS_PATH, index=False)

if __name__ == "__main__":
    appContainer.run(host="0.0.0.0", port=8089)