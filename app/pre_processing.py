# Libraries
import os
import pickle
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer

# Custom
from app.variables import MODEL_FEATURES


class Preprocessor:
    """
    A class to handle data preprocessing, including handling missing values
    using various imputation methods.

    Args:
        imputation_method (str): The imputation strategy ('mean', 'median', or 'constant').
        imputation_value (any, optional): The value to use when strategy is 'constant'. Ignored for other strategies.
        is_training (bool): Flag to indicate whether the preprocessing is for training or inference.
    """

    def __init__(self, is_training=True):
        # Start the classes variables
        self.is_training = is_training
        self.imputer_path = 'models/imputer.pkl'

        # Initialize or load the imputer
        self.imputer = SimpleImputer(strategy='mean') if is_training else self._load_imputer()

    def _load_imputer(self):
        """
        Load a pre-trained imputer from disk.
        """
        # Check if exists
        if not os.path.exists(self.imputer_path):
            raise FileNotFoundError(f"No saved imputer found at {self.imputer_path}. Ensure it exists.")
        # Load the model
        return pickle.load(open(self.imputer_path, "rb"))

    def _save_imputer(self):
        """
        Save the trained imputer to disk for later use.
        """
        with open(self.imputer_path, "wb") as f:
            pickle.dump(self.imputer, f)

    def apply_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data, handling both single samples and batches.

        Args:
            data (pd.DataFrame): Data to preprocess (1 or more rows).

        Returns:
            pd.DataFrame: Preprocessed data with missing values imputed.
        """
        # Check imputer health
        if self.imputer is None:
            raise RuntimeError("Imputer is not initialized. Ensure it is trained or loaded correctly.")

        # Fit the imputer if it is training phase, otherwise will be loaded
        if self.is_training:
            self.imputer.fit(data[MODEL_FEATURES])
            self._save_imputer()
        
        # Apply data transformation
        return self.imputer.transform(data)