'''
Plik zawiera funkcje pomocnicze do treningu modeli LSTM i przygotowania do nich zbiorów danych.
'''

import shap
import itertools
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, InputLayer
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.essentials import *
from scripts.plots import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score
from scipy.special import expit

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

    if len(vector1) != len(vector2):
        raise ValueError("Input vectors must have the same length.")

    matrix = np.zeros((2, 2), dtype=int)

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
    scale = mean_absolute_error(y_train[1:], y_train[:-1]) 
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
    }

    metrics_df = pd.DataFrame(metrics_dict, index=['Train', 'Test', 'Validation'])
    return metrics_df

def calculate_metrics_2(y_true, y_pred):
    """
    Calculate common metrics for time series predictions.

    :param y_true: True values of the target variable
    :param y_pred: Predicted values of the target variable
    :return: Dictionary with calculated metrics
    """
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),     
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),          
        'R2 Score': r2_score(y_true, y_pred),              
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate common metrics for classification tasks.

    :param y_true: True values of the target variable
    :param y_pred: Predicted values of the target variable
    :return: Dictionary with calculated metrics
    """

    y_pred_probabilities = expit(y_pred)  # Convert regression scores to probabilities
    y_true_binary = (y_true > 0).astype(int)  # Convert true values to binary classes
    y_pred_binary = (y_pred > 0).astype(int)  # Convert true values to binary classes

    return {
        'Accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'Precision': precision_score(y_true_binary, y_pred_binary),
        'Recall': recall_score(y_true_binary, y_pred_binary),
        'F1 Score': f1_score(y_true_binary, y_pred_binary),
        'AUC': roc_auc_score(y_true_binary, y_pred_probabilities)
    }

def train_model_from_params(params, X, y, X_val, y_val):
    """
    Train a model using the specified parameters.

    Parameters:
    - params (dict): Dictionary of parameters.
    - X, y: Training data.
    - X_val, y_val: Validation data.

    Returns:
    - tf.keras.Model: Trained model.
    """

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

    model.fit(X, y, epochs=350, validation_data=(X_val, y_val),
                            callbacks=[early_stopping])

    return model

class CustomAUCMetric(tf.keras.metrics.Metric):
    '''
    Custom AUC metric for transforming prediction to binary classification.
    '''
    def __init__(self, name="custom_auc", **kwargs):
        super(CustomAUCMetric, self).__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC(from_logits=True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary = K.cast(K.greater(y_true, 0), 'float32')
        y_pred_binary = K.cast(y_pred, 'float32')
        self.auc.update_state(y_true_binary, y_pred_binary, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_states()

def c_f1_score(y_true, y_pred):
    csm = count_signs_matrix(y_true, y_pred)
    return (3*csm[1][1]/(csm[1][0]+csm[1][1]+0.001) + 2*csm[1][1]/(csm[0][1]+csm[1][1]+0.001))/5

def custom_f1_metric(y_true, y_pred):
    '''
    Custom F1 metric for transforming prediction to binary classification.
    '''
    # Convert y_pred to the same type as y_true
    y_pred = K.cast(y_pred, y_true.dtype)

    # Transform regression outputs to 1 and -1 based on sign
    y_true_binary = K.cast(K.greater(y_true, 0), 'float32')
    y_pred_binary = K.cast(K.greater(y_pred, 0), 'float32')

    # Calculate F1 score
    # Since F1 score is a metric from sklearn, we use tf.py_function to wrap it
    return tf.py_function(c_f1_score, (y_true_binary, y_pred_binary), tf.double)

def create_model(trial, input_shape):
    """
    Create a model with hyperparameters suggested by Optuna.

    Parameters:
    - trial (optuna.Trial): Optuna trial object.
    - input_shape (tuple): Shape of the input data.

    Returns:
    - tf.keras.Model: Model with suggested hyperparameters.
    """

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
        model.add(LSTM(units=trial.suggest_int(f'lstm_units_{i}', 8, 64),
                       return_sequences=(i < n_layers - 1)))

    model.add(Dense(units=trial.suggest_int('dense_units', 4, 32), activation='relu'))
    model.add(Dense(units=1))

    # Compile model
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error',metrics=[CustomAUCMetric()])

    return model

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
    """
    Create and train a model with the specified parameters.

    :param X: Training data.
    :param y: Training labels.
    :param X_val: Validation data.
    :param y_val: Validation labels.
    :param n_epochs: Number of epochs to train for.
    :param lstm_units: Number of LSTM units.
    :param dense_units: Number of units in the Dense layer.
    :param patience: Patience for early stopping.
    :return: The trained model and the training history.
    """
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

    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(X, y, epochs=n_epochs, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

    return model, history

def window_dataset(df, target_column, window_size):
    """
    Create a windowed dataset from a DataFrame.

    :param df: The input DataFrame.
    :param target_column: The name of the target column.
    :param window_size: The size of the window.
    :return: A tuple of (X, y) where X is a 3D array of shape (samples, window_size, num_features) and y is a 1D array
    of shape (samples,).
    """

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

def run_multiple_models(train_set,val_set,cols_subset,n_epochs=100, target_cols= ['target_5'], window_size = 10):
    """
    Run multiple models with different subsets of features and return the results.

    :param train_set: The training set.
    :param val_set: The validation set.
    :param cols_subset: A list of lists of column names to use in each model.
    :param n_epochs: Number of epochs to train for.
    :param target_cols: List of target columns.
    :return: A DataFrame containing the results.
    """
        
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

        #create df as checkpoint
        if (idx%15 == 0):
            df = pd.DataFrame({'cols_used': cols_subset[:idx+1], 'mse_val': mse_val, 'mse_train': mse_train,
                                'aggregated_shap':aggregated_shap})
            df.to_csv('df_shap'+str(idx)+'.csv')
    df = pd.DataFrame({'cols_used': cols_subset, 'mse_val': mse_val, 'mse_train': mse_train,
                            'aggregated_shap':aggregated_shap})
    df.to_csv('df_shap_final.csv')

    return df

def make_model(X, y, n_epochs, lstm_units=32,dense_units=8):
    """
    Create and train a model with the specified parameters.

    :param X: Training data.
    :param y: Training labels.
    :param n_epochs: Number of epochs to train for.
    :param lstm_units: Number of LSTM units.
    :param dense_units: Number of units in the Dense layer.
    :return: The trained model and the training history.
    """
    
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

def get_x_y(df,
            cols_used,
            target_cols= ['target_7'],
            window_size=14,
            is_btc = False):
    """
    Create X and y arrays for training, validation, and testing.

    :param df: The input DataFrame.
    :param cols_used: List of column names to use as features.
    :param target_cols: List of target columns.
    :param window_size: The size of the window.
    :param is_btc: Whether to use options for BTC dataset.
    :return: A tuple of (X, y) where X is a 3D array of shape (samples, window_size, num_features) and y is a 1D array
    of shape (samples,).
    """

    if is_btc:
        X_full, y_full = window_dataset(df[list(cols_used)+target_cols],  target_cols[0], window_size)
        X, y, X_val, y_val, X_test, y_test = X_full[:946-window_size], y_full[:946-window_size],\
                                            X_full[(946-window_size):(946-window_size+56)], y_full[(946-window_size):(946-window_size+56)], \
                                            X_full[(946-window_size+56):-7], y_full[(946-window_size+56):-7] 
    else:
        X_full, y_full = window_dataset(df[list(cols_used)+target_cols],  target_cols[0], window_size)
        X, y, X_val, y_val, X_test, y_test = X_full[:656-window_size], y_full[:656-window_size],\
                                        X_full[(656-window_size):(656-window_size+40)], y_full[(656-window_size):(656-window_size+40)], \
                                        X_full[(656-window_size+40):-5], y_full[(656-window_size+40):-5]

    y_base0_val = [0 for i in range((y_val.shape[0]))]
    y_base0_train = [0 for i in range((y.shape[0]))]
    y_base0_test = [0 for i in range(y_test.shape[0])]

    return X,y,X_val,y_val,X_test,y_test,y_base0_train,y_base0_val,y_base0_test


def predict_from_lstm(model,data, output_path, prefix, cols = 'fin'):
    """
    Predict the target variable using a trained LSTM model.

    :param model: The trained model.
    :param data: The input DataFrame.
    :param output_path: Path to save the predictions.
    :param prefix: Prefix of the dataset.
    :param cols: Type of features to use.
    """

    if prefix == "BA":

        window_size = 10

        target_cols = ['target_5']

        if cols == 'fin':

            cols_used = [
                    'norm_rsi_14', 'norm_slowk_14', 'minmax_daily_variation', 'minmax_BA_Volume',
                    #'mean_influential', 'mean_trustworthy', 'finbert_Score', 'bart_Score'
                    ]
        elif cols == 'news':

            cols_used = [
                    #'norm_rsi_14', 'norm_slowk_14', 'minmax_daily_variation', 'minmax_BA_Volume',
                    'mean_influential', 'mean_trustworthy', 'finbert_Score', 'bart_Score'
                    ]
            
        else:

            cols_used = [
                    'norm_rsi_14', 'norm_slowk_14', 'minmax_daily_variation', 'minmax_BA_Volume',
                    'mean_influential', 'mean_trustworthy', 'finbert_Score', 'bart_Score'
                    ]

        step = 5

    elif prefix == "TSLA":

        window_size = 10

        target_cols = ['target_5']

        if cols == 'fin':

            cols_used = [
                        'minmax_low_norm', 'minmax_high_norm', 'norm_rsi_gspc_14', 'norm_slowk_14' ,
                        #'vader_Score', 'bart_Score', 'mean_influential', 'finbert_Score', 'mean_trustworthy'
                        ]
            
        elif cols == 'news':

            cols_used = [
                        #'minmax_low_norm', 'minmax_high_norm', 'norm_rsi_gspc_14', 'norm_slowk_14' ,
                        'vader_Score', 'bart_Score', 'mean_influential', 'finbert_Score', 'mean_trustworthy'
                        ]
            
        else:

            cols_used = [
                        'minmax_low_norm', 'minmax_high_norm', 'norm_rsi_gspc_14', 'norm_slowk_14' ,
                        'vader_Score', 'bart_Score', 'mean_influential', 'finbert_Score', 'mean_trustworthy'
                        ]

        step = 5

    elif prefix == "NFLX":

        window_size = 10

        target_cols = ['target_5']


        if cols == 'fin':

            cols_used = [
                    'norm_rsi_gspc_14', 'norm_rsi_14','norm_slowk_14', 'minmax_high_norm', 'log_return_1',
                    #'finbert_Score', 'bart_Score', 'vader_Score','mean_influential', 'mean_trustworthy'
                    ]
            
        elif cols == 'news':

            cols_used = [
                    #'norm_rsi_gspc_14', 'norm_rsi_14','norm_slowk_14', 'minmax_high_norm', 'log_return_1',
                    'finbert_Score', 'bart_Score', 'vader_Score','mean_influential', 'mean_trustworthy'
                    ]
        
        else:
                
                cols_used = [
                        'norm_rsi_gspc_14', 'norm_rsi_14','norm_slowk_14', 'minmax_high_norm', 'log_return_1',
                        'finbert_Score', 'bart_Score', 'vader_Score','mean_influential', 'mean_trustworthy'
                        ]
                
        step = 5

    elif prefix == "BTC-USD":

        window_size = 14

        target_cols = ['target_7']

        if cols == 'fin':

            cols_used = [
                        'minmax_BTC-USD_Volume', 'norm_rsi_14', 'norm_slowk_14', 'norm_rsi_gspc_14', 'minmax_daily_variation',
                        #'finbert_Score', 'vader_Score', 'mean_influential'
                        ]
            
        elif cols == 'news':
                
            cols_used = [
                        #'minmax_BTC-USD_Volume', 'norm_rsi_14', 'norm_slowk_14', 'norm_rsi_gspc_14', 'minmax_daily_variation',
                        'finbert_Score', 'vader_Score', 'mean_influential'
                        ]
        
        else:
                    
            cols_used = [
                        'minmax_BTC-USD_Volume', 'norm_rsi_14', 'norm_slowk_14', 'norm_rsi_gspc_14', 'minmax_daily_variation',
                        'finbert_Score', 'vader_Score', 'mean_influential'
                        ]

        step = 7

    else:
        print("Prefix not found")
        exit()

    X, _ = window_dataset(data[list(cols_used)+target_cols], target_cols[0], window_size)

    y_pred = model.predict(X)

    dates = data['Date']

    if step == 5:

        next_dates = pd.bdate_range(start=max(dates), periods=step+1)[1:].strftime('%Y-%m-%d')

    else:

        next_dates = pd.date_range(start=max(dates), periods=step+1)[1:].strftime('%Y-%m-%d')

    dates_extended = pd.Series(dates).append(pd.Series(next_dates))

    df_pred = pd.DataFrame(data= {'pred': y_pred.flatten(), 'Date': dates_extended[(window_size+step):]})
    df_pred.to_csv(output_path, index=False)
