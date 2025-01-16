# Libraries
import pandas as pd
from numpy import nan

# Custom
from app.variables import MODEL_FEATURES, TARGET_LABEL
from app.variables import LIM_MIN_AGE, LIM_MAX_AGE, LIM_MIN_INCOME, LIM_MIN_LOAN, LIM_MIN_OPEN_ACCOUNTS

class DataValidator:
    def __init__(self):
        pass

    def check_feature_values(self, value, lim_min=None, lim_max=None):
        '''
        Checks if a given value is within the specified range defined by minimum and/or maximum limits. 
        Returns the value if it is within the range, otherwise returns NaN.
        Args:
            value (float or int): The feature value to validate.
            lim_min (float or int, optional): The minimum acceptable limit for the value. Default is None (no lower bound).
            lim_max (float or int, optional): The maximum acceptable limit for the value. Default is None (no upper bound).
        Returns:
            float or int: The original value if it is within the range.
            nan: If the value falls outside the specified limits.
        '''
        # Deal with different scenarios of limit application
        if lim_min is None and lim_max is None:
            return value
        elif lim_min is not None and lim_max is None:
            return nan if value < lim_min else value
        elif lim_min is None and lim_max is not None:
            return nan if value > lim_max else value
        else:
            return nan if value < lim_min or value > lim_max else value

    def validate(self, data: pd.DataFrame):
        """
        Validate the input data (single sample or batch).
        Args:
            data (pd.DataFrame): Input data to validate.
        Returns:
            pd.DataFrame: Validated data with invalid values replaced with NaN.
        """
        # Ensure all required columns are present
        missing_columns = [col for col in MODEL_FEATURES + TARGET_LABEL if col not in data.columns]
        if len(missing_columns) > 0:
            raise ValueError(f"Missing columns in input data: {missing_columns}")

        # Replace invalid values with NaN
        data['Age'] = data['Age'].apply(lambda x: self.check_feature_values(x, LIM_MIN_AGE, LIM_MAX_AGE))
        data['Annual_Income'] = data['Annual_Income'].apply(lambda x: self.check_feature_values(x, LIM_MIN_INCOME))
        data['Loan_Amount'] = data['Loan_Amount'].apply(lambda x: self.check_feature_values(x, LIM_MIN_LOAN))
        data['Number_of_Open_Accounts'] = data['Number_of_Open_Accounts'].apply(lambda x: self.check_feature_values(x, LIM_MIN_OPEN_ACCOUNTS))

        # Return final result
        return data