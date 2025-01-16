# Libraries
import os
import pickle
import pandas as pd
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score

# Custom
from app.pre_processing import Preprocessor
from app.data_validation import DataValidator
from app.variables import MODEL_FEATURES, TARGET_LABEL

# Paths
TRAINING_DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/classifier.pkl"

def train_pipeline():
    # Load the training data
    data = pd.read_csv(TRAINING_DATA_PATH)

    # Validate feature and label data
    data = DataValidator().validate(data)

    # Split features and target
    x = data[MODEL_FEATURES].copy()
    y = data[TARGET_LABEL]

    # Preprocess + Train with K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1206)
    kmetrics = list()

    # Start Split - Preprocessing - Training - Validation Metrics
    for train_idx, valid_idx in kfold.split(x, y):
        # Split the data
        x_train, x_valid = x.loc[train_idx, :], x.loc[valid_idx, :]
        y_train, y_valid = y.loc[train_idx, :], y.loc[valid_idx, :]

        # Convert label to 1-D array (avoid warnings)
        y_train, y_valid = y_train.values.ravel(), y_valid.values.ravel()

        # Apply preprocessing for trainset (fit / save & transform)
        x_train = Preprocessor(is_training=True).apply_preprocessing(x_train)
        x_valid = Preprocessor(is_training=False).apply_preprocessing(x_valid)

        # Train the model
        mdl = RandomForestClassifier(n_estimators=120).fit(x_train, y_train)

        # Evaluate on the validation set
        y_pred = mdl.predict(x_valid)
        kmetrics.append({'F1S': f1_score(y_valid, y_pred),
                         'REC': recall_score(y_valid, y_pred),
                         'PRE': precision_score(y_valid, y_pred)})
        
    # Report scores
    f1s = round(100*mean([km['F1S'] for km in kmetrics]), 2)
    rec = round(100*mean([km['REC'] for km in kmetrics]), 2)
    pre = round(100*mean([km['PRE'] for km in kmetrics]), 2)
    print(f'5-Fold Stratified Cross Validation\nF1-Score = {f1s}% | Recall = {rec}% | Precision = {pre}%')

    # Preprocess + Train the model with the complete data
    x = Preprocessor(is_training=True).apply_preprocessing(x)
    y = y.values.ravel()
    mdl = RandomForestClassifier(n_estimators=120).fit(x, y)

    # Save the trained model
    pickle.dump(mdl, open(MODEL_PATH, "wb"))
    
    # Report
    print("Training complete! Model saved.")

if __name__ == "__main__":
    train_pipeline()