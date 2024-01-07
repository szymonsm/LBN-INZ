from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, InputLayer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import shap
import itertools
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, InputLayer, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, auc
from keras.models import load_model
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import optuna
import tensorflow as tf
from scripts.essentials import *
from scripts.plots import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def train_model_from_params(params, X, y, X_val, y_val):
  # Create the model
  model = Sequential()

  # Add Conv1D layer if use_conv is True
  if params['use_conv']:
      model.add(Conv1D(filters=params['conv_filters'],
                      kernel_size=params['conv_kernel_size'],
                      activation='relu',
                      input_shape=(X.shape[1], X.shape[2])))
      model.add(MaxPooling1D(pool_size=2))

  # Add LSTM layers
  for i in range(params['n_layers']):
      model.add(LSTM(units=params[f'lstm_units_{i}'],
                    return_sequences=(i < params['n_layers'] - 1)))

  # Add Dense layer and output layer
  model.add(Dense(units=params['dense_units'], activation='relu'))
  model.add(Dense(units=1))

  # Compile the model
  model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_squared_error',metrics=[CustomAUCMetric()])

  # Train the model
  early_stopping = EarlyStopping(monitor='val_custom_auc', patience=40, mode='max', restore_best_weights=True)

  history = model.fit(X, y, epochs=350, validation_data=(X_val, y_val),
                          callbacks=[early_stopping])

  return model

class CustomAUCMetric(tf.keras.metrics.Metric):
    def __init__(self, name="custom_auc", **kwargs):
        super(CustomAUCMetric, self).__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC(from_logits=True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary = K.cast(K.greater(y_true, 0), 'float32')
        y_pred_binary = K.cast(K.greater(y_pred, 0), 'float32')
        self.auc.update_state(y_true_binary, y_pred_binary, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_states()

def c_f1_score(y_true, y_pred):
    csm = count_signs_matrix(y_true, y_pred)
    #print(csm)
    return (3*csm[1][1]/(csm[1][0]+csm[1][1]+0.001) + 2*csm[1][1]/(csm[0][1]+csm[1][1]+0.001))/5

def custom_f1_metric(y_true, y_pred):
    # Convert y_pred to the same type as y_true
    y_pred = K.cast(y_pred, y_true.dtype)

    # Transform regression outputs to 1 and -1 based on sign
    y_true_binary = K.cast(K.greater(y_true, 0), 'float32')
    y_pred_binary = K.cast(K.greater(y_pred, 0), 'float32')

    # Calculate F1 score
    # Since F1 score is a metric from sklearn, we use tf.py_function to wrap it
    return tf.py_function(c_f1_score, (y_true_binary, y_pred_binary), tf.double)

def custom_auc_metric(y_true, y_pred):
    # Convert y_pred to the same type as y_true
    y_pred = K.cast(y_pred, y_true.dtype)

    # Transform regression outputs to 0 and 1 based on sign
    y_true_sign = K.sign(y_true)
    y_pred_sign = K.sign(y_pred)

    # Use TensorFlow's AUC metric
    return tf.py_function(auc, (y_true_sign, y_pred_sign), tf.double)


def create_model(trial, input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    # Optionally add a convolutional layer before LSTM
    if trial.suggest_categorical('use_conv', [True, False]):
        model.add(Conv1D(filters=trial.suggest_categorical('conv_filters', [32, 64, 128]),
                         kernel_size=trial.suggest_int('conv_kernel_size', 2, 5),
                         activation='relu',
                         strides=1,
                      padding='causal'))
        model.add(MaxPooling1D(pool_size=2))

    # Optuna can suggest the number of LSTM layers and units
    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        model.add(LSTM(units=trial.suggest_categorical(f'lstm_units_{i}', [16, 32, 64]),
                       return_sequences=(i < n_layers - 1)))

    model.add(Dense(units=trial.suggest_categorical('dense_units', [8, 16, 32]), activation='relu'))
    model.add(Dense(units=1))

    # Compile model
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error',metrics=[CustomAUCMetric()])

    return model

def objective(trial):
    model = create_model(trial, input_shape=(X.shape[1], X.shape[2]))

    early_stopping = EarlyStopping(monitor='val_custom_auc', patience=35, mode='max', restore_best_weights=True)

    history = model.fit(X, y, epochs=200, validation_data=(X_val, y_val),
                        callbacks=[early_stopping], verbose=0)

    # Evaluate the model
    val_loss, val_f1 = model.evaluate(X_val, y_val, verbose=0)
    return val_f1

def get_x_y(train_set,
            val_set,
            test_set,
            cols_used,
            target_cols= ['target_5'],
            window_size=10,
            shift_=5):

  X,y = window_dataset(train_set[list(cols_used)+target_cols],  target_cols[0], window_size)
  X_val, y_val = window_dataset(val_set[list(cols_used)+target_cols], target_cols[0], window_size)
  X_test, y_test = window_dataset(test_set[list(cols_used)+target_cols], target_cols[0], window_size)

  y_test = y_test[:-shift_]

  y_base_val = val_set[target_cols[0]].shift(shift_)[window_size:].values
  y_base_test = test_set[target_cols[0]].shift(shift_)[window_size:-shift_].values
  y_base_train = train_set[target_cols[0]].shift(shift_)[window_size:].values
  y_base0_val = [0 for i in range((val_set.shape[0]-window_size))]
  y_base0_train = [0 for i in range((train_set.shape[0]-window_size))]
  y_base0_test = [0 for i in range(len(y_test))]

  return X,y,X_val,y_val,X_test,y_test,y_base_train,y_base_val,y_base_test,y_base0_train,y_base0_val,y_base0_test


def create_unique_subsets(columns, subset_size=5, min_diff=2):
    """
    Create subsets of columns where each subset differs from every other subset by at least 'min_diff' columns.

    :param columns: List of all columns.
    :param subset_size: The size of each subset.
    :param min_diff: Minimum number of different columns between any two subsets.
    :return: A list of unique subsets.
    """
    all_combinations = list(itertools.combinations(columns, subset_size))
    unique_subsets = []

    for combo in all_combinations:
        if all(len(set(combo) - set(subset)) >= min_diff for subset in unique_subsets):
            unique_subsets.append(combo)

    return unique_subsets

def make_model_better(X, y, X_val, y_val, n_epochs, lstm_units=64, dense_units=10, patience=5):
    model = Sequential()
    model.add(InputLayer(input_shape=(X.shape[1], X.shape[2])))

    # LSTM layers
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(LSTM(units=lstm_units, return_sequences=False))

    # Dense layers
    model.add(Dense(units=dense_units))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(X, y, epochs=n_epochs, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

    return model, history

# Example usage:
# model, history = make_model(X_train, y_train, X_val, y_val, n_epochs=50, lstm_units=64, dense_units=10, patience=5)

def window_dataset(df, target_column, window_size):
    X = []
    y = []

    # Iterate over the dataset
    for i in range(len(df) - window_size):
        # Extract the window of data
        window = df[i:i+window_size]

        # Extract the features (X) and target (y)
        X.append(window.drop(columns=[target_column]).values)
        y.append(df[target_column][i+window_size])

    return np.array(X), np.array(y)

def run_multiple_models(train_set,val_set,cols_subset,n_epochs=100):

  window_size = 14
  target_cols = ['target_7']
  mse_val = []
  mse_train = []
  aggregated_shap = []
  for idx,cols_used in enumerate(cols_subset):
    print('-------------------------------')
    print(idx)

    X,y = window_dataset(train_set[list(cols_used)+target_cols],  target_cols[0], window_size)
    X_val, y_val = window_dataset(val_set[list(cols_used)+target_cols], target_cols[0], window_size)

    model,history = make_model(X,y, X_val, y_val, n_epochs)

    mse_val.append(min(history.history['val_loss']))
    mse_train.append(min(history.history['loss']))

    explainer = shap.DeepExplainer(model, X)
    shap_values = explainer.shap_values(X_val)

    aggregated_shap.append(np.mean(np.abs(shap_values[0]), axis=0))

    #create df
    if (idx%15 == 0):
      df = pd.DataFrame({'cols_used': cols_subset[:idx+1], 'mse_val': mse_val, 'mse_train': mse_train,
                        'aggregated_shap':aggregated_shap})
      df.to_csv('df_shap'+str(idx)+'.csv')
  df = pd.DataFrame({'cols_used': cols_subset, 'mse_val': mse_val, 'mse_train': mse_train,
                        'aggregated_shap':aggregated_shap})
  df.to_csv('df_shap_final.csv')

  return df

def parse_cols_used(cols_string):
    """
    Parses the column names used in the model from the string representation.

    Args:
    cols_string (str): A string representing the tuple of column names.

    Returns:
    list: A list of column names.
    """
    # Remove the outer brackets and split the string into column names
    cols = cols_string.strip("()").split(", ")
    # Remove quotes and extra characters from column names
    cleaned_cols = [col.strip("'") for col in cols]
    return cleaned_cols

def parse_shap_values(shap_string):
    """
    Parses a string representation of a list of lists of SHAP values into a Python list of lists.

    Args:
    shap_string (str): A string representing a list of lists of SHAP values.

    Returns:
    list of lists: A list of lists of SHAP values.
    """
    # Remove the outer brackets and split the string into rows, handling newline characters
    rows = shap_string.strip('[]').split('\n')

    # Parse each row into a list of floats
    parsed_rows = [list(map(float, row.strip(' []').split())) for row in rows]

    return parsed_rows

def make_model(X, y, n_epochs, lstm_units=32,dense_units=8):
    tf.keras.backend.clear_session()

    model = Sequential()
    model.add(InputLayer(input_shape=(X.shape[1], X.shape[2])))

    # LSTM layers
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(LSTM(units=lstm_units, return_sequences=False))

    # Dense layers
    model.add(Dense(units=dense_units))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X, y, epochs=n_epochs,
                        verbose=0)

    return model, history

def plot_mean_feature_importance(shap_values, feature_names,max_=False):
    """
    Plot the mean importance of each feature across all time steps.

    :param shap_values: A 3D array of SHAP values with shape (samples, window_size, num_features).
    :param feature_names: List of feature names.
    """
    if max_:
    # Aggregate SHAP values across all samples and time steps
      aggregated_shap = np.max(np.abs(shap_values), axis=(0))
    else:
      aggregated_shap = np.mean(np.abs(shap_values), axis=(0))


    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, aggregated_shap)
    plt.title('Mean Feature Importance Across All Time Steps')
    plt.ylabel('Mean SHAP Value')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
