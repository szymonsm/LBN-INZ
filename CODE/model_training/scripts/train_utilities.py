
def calculate_lagged_correlation(col1, col2, df, lag=14):
    """
    Calculate time-series correlation between two columns across a specified time lag.

    Parameters:
    - col1 (str): Name of the first column.
    - col2 (str): Name of the second column.
    - df (pd.DataFrame): The input DataFrame.
    - lag (int): Maximum lag for which to calculate correlation.

    Returns:
    - pd.Series: Series containing correlations for each lag.
    """
    # Create an empty DataFrame to store correlations
    correlations = pd.DataFrame(columns=['Lag', 'Correlation'])

    # Iterate over lags and calculate correlation for each lag shift
    for i in range(1, lag + 1):
        shifted_col2 = df[col2].shift(i)
        correlation = df[col1].corr(shifted_col2)

        # Save lag and correlation in the DataFrame
        correlations = correlations.append({'Lag': i, 'Correlation': correlation}, ignore_index=True)

    return correlations


def count_signs_matrix(vector1, vector2):
    """
    Count occurrences of signs in two vectors and return a 2x2 matrix.

    Parameters:
    - vector1 (array-like): First vector.
    - vector2 (array-like): Second vector.

    Returns:
    - np.ndarray: 2x2 matrix representing sign occurrences.
    """
    # Ensure the input vectors have the same length
    if len(vector1) != len(vector2):
        raise ValueError("Input vectors must have the same length.")

    # Initialize the matrix with zeros
    matrix = np.zeros((2, 2), dtype=int)

    # Count sign occurrences
    for val1, val2 in zip(vector1, vector2):
        if val1 > 0 and val2 > 0:
            matrix[0, 0] += 1
        elif val1 > 0 and val2 < 0:
            matrix[0, 1] += 1
        elif val1 < 0 and val2 > 0:
            matrix[1, 0] += 1
        elif val1 < 0 and val2 < 0:
            matrix[1, 1] += 1

    return matrix


def split_data(df, date_col, split_date_val, split_date_test, start_date_train=None):
    """
    Split the DataFrame into train, validation, and test sets based on specified dates.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - date_col (str): Column name for the date.
    - split_date_val (str): Date for splitting between validation and test sets.
    - split_date_test (str): Date for splitting between train and validation sets.
    - start_date_train (str or None): Start date for training data. If None, the earliest date in the DataFrame is used.

    Returns:
    - pd.DataFrame, pd.DataFrame, pd.DataFrame: Train, validation, and test sets.
    """
    # Ensure the DataFrame is sorted by date
    df.sort_values(by=date_col, inplace=True)

    # Determine the start date for training data
    if start_date_train is None:
        start_date_train = df[date_col].min()

    # Split the data into train, validation, and test sets
    train_data = df[(df[date_col] >= start_date_train) & (df[date_col] < split_date_val)]
    val_data = df[(df[date_col] >= split_date_val) & (df[date_col] < split_date_test)]
    test_data = df[df[date_col] >= split_date_test]

    return train_data, val_data, test_data

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    scale = mean_absolute_error(y_train[1:], y_train[:-1])  # Scale is the MAE of the training set differences
    return mean_absolute_error(y_true, y_pred) / scale

def calculate_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_val, y_pred_val):
    """
    Calculate common metrics for time series predictions.

    Parameters:
    - y_true_train, y_pred_train: True and predicted values for the training set.
    - y_true_test, y_pred_test: True and predicted values for the test set.
    - y_true_val, y_pred_val: True and predicted values for the validation set.

    Returns:
    - pd.DataFrame: DataFrame with calculated metrics for each set (train, test, val).
    """
    metrics_dict = {
        'MAE': [mean_absolute_error(y_true_train, y_pred_train),
                mean_absolute_error(y_true_test, y_pred_test),
                mean_absolute_error(y_true_val, y_pred_val)],
        'MSE': [mean_squared_error(y_true_train, y_pred_train),
                mean_squared_error(y_true_test, y_pred_test),
                mean_squared_error(y_true_val, y_pred_val)],
        'RMSE': [mean_squared_error(y_true_train, y_pred_train, squared=False),
                 mean_squared_error(y_true_test, y_pred_test, squared=False),
                 mean_squared_error(y_true_val, y_pred_val, squared=False)],
        'R2 Score': [r2_score(y_true_train, y_pred_train),
                     r2_score(y_true_test, y_pred_test),
                     r2_score(y_true_val, y_pred_val)],
        'MAPE': [mean_absolute_percentage_error(y_true_train, y_pred_train),
                 mean_absolute_percentage_error(y_true_test, y_pred_test),
                 mean_absolute_percentage_error(y_true_val, y_pred_val)],
        'MASE': [mean_absolute_scaled_error(y_true_train, y_pred_train, y_true_train),
                 mean_absolute_scaled_error(y_true_test, y_pred_test, y_true_train),
                 mean_absolute_scaled_error(y_true_val, y_pred_val, y_true_train)]
        # Add more metrics as needed
    }

    metrics_df = pd.DataFrame(metrics_dict, index=['Train', 'Test', 'Validation'])
    return metrics_df