import pandas as pd
import optuna
from neuralforecast.models import NBEATSx, NHITS, TFT
from neuralforecast import NeuralForecast
from functools import partial
import json
import os
from sklearn.metrics import mean_squared_error
from datetime import datetime
from neuralforecast.losses.pytorch import MSE
import numpy as np

# Initialize model names
NAME_MODEL = {"NBEATSx": NBEATSx, "NHITS": NHITS, "TFT": TFT}

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

def create_trial_params_tft(trial):
    """
    Function to create trial parameters for TFT model.

    :param trial: optuna trial object

    :return: dictionary of parameters
    """
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 12, 16, 20, 24, 28, 32])
    model_params = {"learning_rate": learning_rate, "hidden_size": hidden_size}

    return model_params


def create_trial_params_nbeats(trial):
    """
    Function to create trial parameters for NBEATSx model.

    :param trial: optuna trial object

    :return: dictionary of parameters
    """
    
    n_blocks_season = trial.suggest_int('n_blocks_season', 1, 3)
    n_blocks_trend = trial.suggest_int('n_blocks_trend', 1, 3)
    n_blocks_identity = trial.suggest_int('n_blocks_ident', 1, 3)
    
    mlp_units_n = trial.suggest_categorical('mlp_units', [32, 64, 128])
    num_hidden = trial.suggest_int('num_hidden', 1, 3)
    
    n_harmonics = trial.suggest_int('n_harmonics', 1, 5)
    n_polynomials = trial.suggest_int('n_polynomials', 1, 5)
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
    
    n_blocks = [n_blocks_season, n_blocks_trend, n_blocks_identity]
    mlp_units=[[mlp_units_n, mlp_units_n]]*num_hidden
    
    model_params = {
      "stack_types": ['seasonality', 'trend', 'identity'], "mlp_units": mlp_units, "n_blocks": n_blocks,
      "n_harmonics": n_harmonics, "n_polynomials": n_polynomials, "learning_rate": learning_rate}
    
    return model_params

def create_trial_params_nhits(trial):
    """
    Function to create trial parameters for NHITS model.

    :param trial: optuna trial object

    :return: dictionary of parameters
    """    
    n_blocks1 = trial.suggest_int('n_blocks1', 1, 3)
    n_blocks2 = trial.suggest_int('n_blocks2', 1, 3)
    n_blocks3 = trial.suggest_int('n_blocks3', 1, 3)
    
    mlp_units_n = trial.suggest_categorical('mlp_units', [32, 64, 128])
    
    n_pool_kernel_size1 = trial.suggest_int('n_pool_kernel_size1', 1, 3)
    n_pool_kernel_size2 = trial.suggest_int('n_pool_kernel_size2', 1, 3)
    n_pool_kernel_size3 = trial.suggest_int('n_pool_kernel_size3', 1, 3)
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)    
    
    n_blocks = [n_blocks1, n_blocks2, n_blocks3]
    mlp_units=[[mlp_units_n, mlp_units_n]]*3
    n_pool_kernel_size = [n_pool_kernel_size1, n_pool_kernel_size2, n_pool_kernel_size3]
    stack_types = ['identity', 'identity', 'identity']
    
    model_params = {"mlp_units": mlp_units, "n_blocks": n_blocks, "stack_types": stack_types,
    "learning_rate": learning_rate, "n_pool_kernel_size": n_pool_kernel_size}
    
    return model_params



def pipeline_train_predict(models, train_set, val_set, horizon, loss_func, model_name):
    """
    Function to generalize training process between different NeuralForecast models.

    :param models: list of NeuralForecast models with their parameters
    :param train_set: training set
    :param val_set: validation set
    :param metric_function: metric function to evaluate the model
    :param model_name: name of the model - predictions are in column named after model

    :return: tuple of metric value on validation set and predictions on validation set
    """
    predictions = None
    val_set_begin = val_set.iloc[0]['ds']

    train_set_tmp = train_set
    last_index = horizon
    val_set_tmp = val_set.iloc[:last_index]
    modelID = 0
    while train_set_tmp.iloc[0]['ds'] < val_set_begin and len(val_set_tmp) == horizon:
        predictions_tmp = train(models, train_set_tmp, val_set_tmp, model_name)
        predictions_tmp['modelID'] = modelID
        modelID += 1
        predictions = predictions.append(predictions_tmp) if predictions is not None else predictions_tmp

        train_set_tmp = train_set_tmp.append(val_set_tmp)
        val_set_tmp = val_set.iloc[last_index:last_index+horizon]
        last_index = last_index + horizon
    predictions['sequenceID'] = predictions.index
    predictions.reset_index(drop=True, inplace=True)
    predictions = predictions[predictions['y'].notnull()]
    loss = loss_func(predictions['y'], predictions[f'{model_name}'])

    return loss, predictions


def train(models: list, train_set: pd.DataFrame, val_set: pd.DataFrame, model_name: str):
    """
    Function to generalize training process between different NeuralForecast models.

    :param models: list of NeuralForecast models with their parameters
    :param train_set: training set
    :param val_set: validation set
    :param metric_function: metric function to evaluate the model
    :param model_name: name of the model - predictions are in column named after model

    :return: tuple of metric value on validation set and predictions on validation set
    """
    model = NeuralForecast(models=models, freq='D')
    model.fit(train_set)

    p =  model.predict().reset_index()
    p = p.merge(val_set[['ds','unique_id', 'y']].reset_index(), on=['ds', 'unique_id'], how='left')

    return p

def create_trial_params(trial, model_name: str):
    """
    Function to create trial parameters for different NeuralForecast models.

    :param trial: optuna trial object
    :param model_name: name of the model

    :return: dictionary of parameters
    """
    if model_name == "TFT":
      return create_trial_params_tft(trial)
    if model_name == "NBEATSx":
      return create_trial_params_nbeats(trial)
    if model_name == "NHITS":
      return create_trial_params_nhits(trial)     


def objective(trial, train_set, val_set, loss, model_name: str, horizon: int, hist_exog_list: list, max_steps: int, random_seed: int, loss_func: callable, scaler_type = 'standard'):
    """
    Function to optimize NeuralForecast models.

    :param trial: optuna trial object
    :param train_set: training set
    :param val_set: validation set
    :param loss: loss function
    :param model_name: name of the model
    :param horizon: prediction horizon
    :param hist_exog_list: list of exogenous variables
    :param max_steps: maximum number of training steps
    :param random_seed: random seed
    :param loss_func: metric function to evaluate the model
    :param scaler_type: type of scaler

    :return: metric value on validation set
    """
    model_params = create_trial_params(trial, model_name)
    models = [NAME_MODEL[model_name](h=horizon, input_size=2*horizon, loss=loss, hist_exog_list=hist_exog_list, random_seed=random_seed, scaler_type=scaler_type, max_steps=max_steps, **model_params)]

    loss, _ = pipeline_train_predict(models, train_set, val_set, horizon, loss_func, model_name)
    return loss

def eval_score(y_true, y_pred):
    """
    Function to evaluate model performance.

    :param y_true: true values
    :param y_pred: predicted values

    :return: score
    """
    csm = count_signs_matrix(y_true.reset_index(drop=True), y_pred.reset_index(drop=True))
    return (3*csm[1][1]/(csm[1][0]+csm[1][1]+0.0001) + 2*csm[1][1]/(csm[0][1]+csm[1][1]+0.0001))/5

class HorizonTrainer:
    """
    Class to train models with different horizons.
    """
    def __init__(self, prefix, mode, model_type, final_columns, train_set_all, val_set_all, test_set_all, horizon, max_steps, scaler_type, loss_func, loss, n_trials, random_seed, timestamp, target):
        """
        :param prefix: prefix of the model
        :param mode: mode of the model
        :param model_type: type of the model
        :param final_columns: list of exogenous variables
        :param train_set_all: training set
        :param val_set_all: validation set
        :param test_set_all: test set
        :param horizon: prediction horizon
        :param max_steps: maximum number of training steps
        :param scaler_type: type of scaler
        :param loss_func: metric function to evaluate the model
        :param loss: loss function
        :param n_trials: number of trials for optuna
        :param random_seed: random seed
        :param timestamp: timestamp
        :param target: target variable
        """
        self.prefix = prefix
        self.model_type = model_type
        self.mode = mode
        self.final_columns = final_columns
        self.train_set_all = train_set_all
        self.train_set = train_set_all
        self.val_set_all = val_set_all
        self.val_set = val_set_all
        self.test_set_all = test_set_all
        self.test_set = test_set_all
        self.horizon = horizon
        self.max_steps = max_steps
        self.scaler_type = scaler_type
        self.loss_func = loss_func
        self.loss = loss
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.timestamp = timestamp
        self.target = target

    def __prepare_sets__(self):
        """
        Function to prepare sets for training.

        :return: None
        """

        self.train_set_all['DateGen'] = pd.date_range(start='2021-01-01', end='2023-11-26', freq='D')[:len(self.train_set)]
        valid_start_date = self.train_set['DateGen'].max() + pd.DateOffset(days=1)
        self.val_set_all['DateGen'] = pd.date_range(start=valid_start_date, end='2023-11-26', freq='D')[:len(self.val_set)]
        test_start_date = self.val_set['DateGen'].max() + pd.DateOffset(days=1)
        self.test_set_all['DateGen'] = pd.date_range(start=test_start_date, end='2023-11-26', freq='D')[:len(self.test_set)]
        self.train_set = self.train_set_all[['DateGen', f'{self.target}'] + self.final_columns]
        self.val_set = self.val_set_all[['DateGen', f'{self.target}'] + self.final_columns]
        self.test_set = self.test_set_all[['DateGen', f'{self.target}'] + self.final_columns]
        self.train_set.rename(columns={'DateGen':'ds', f'{self.target}':'y'}, inplace=True)
        self.val_set.rename(columns={'DateGen':'ds', f'{self.target}':'y'}, inplace=True)
        self.test_set.rename(columns={'DateGen':'ds', f'{self.target}':'y'}, inplace=True)
        self.train_set['unique_id'] = self.prefix
        self.val_set['unique_id'] = self.prefix
        self.test_set['unique_id'] = self.prefix

    def train(self):
        """
        Function to train models. Saves predictions, parameters and loss to created directories.

        :return: None
        """
        self.__prepare_sets__()
        if self.loss_func == eval_score:
            study = optuna.create_study(direction='maximize')
        else:
            study = optuna.create_study(direction='minimize')
        study.optimize(partial(objective, train_set=self.train_set, val_set=self.val_set, loss=self.loss, model_name=self.model_type, horizon=self.horizon, hist_exog_list=self.final_columns, max_steps=self.max_steps, random_seed=self.random_seed, loss_func=self.loss_func, scaler_type=self.scaler_type), n_trials=self.n_trials)
        prms = study.best_trial.params

        if self.model_type == 'NBEATSx':
            n_blocks = [prms['n_blocks_season'], prms['n_blocks_trend'], prms['n_blocks_ident']]
            mlp_units=[[prms['mlp_units'], prms['mlp_units']]]*prms['num_hidden']
            params = {
                'h': self.horizon,
                'loss': self.loss,
                'max_steps': self.max_steps,
                'hist_exog_list': self.final_columns,
                'input_size': 2*self.horizon,
                'stack_types': ['seasonality', 'trend', 'identity'],
                'mlp_units': mlp_units,
                'n_blocks': n_blocks,
                'learning_rate': prms['learning_rate'],
                'n_harmonics': prms['n_harmonics'],
                'n_polynomials': prms['n_polynomials'],
                'scaler_type': self.scaler_type,
                'random_seed': self.random_seed
            }
            models = [NBEATSx(**params)]

        elif self.model_type == 'NHITS':
            n_blocks = [prms['n_blocks1'], prms['n_blocks2'], prms['n_blocks3']]
            mlp_units=[[prms['mlp_units'], prms['mlp_units']]]*3
            n_pool_kernel_size = [prms['n_pool_kernel_size1'], prms['n_pool_kernel_size2'], prms['n_pool_kernel_size3']]
            params = {
                'h': self.horizon,
                'loss': self.loss,
                'max_steps': self.max_steps,
                'hist_exog_list': self.final_columns,
                'input_size': 2*self.horizon,
                'stack_types': ['identity', 'identity', 'identity'],
                'mlp_units': mlp_units,
                'n_blocks': n_blocks,
                'learning_rate': prms['learning_rate'],
                'n_pool_kernel_size': n_pool_kernel_size,
                'scaler_type': self.scaler_type,
                'random_seed': self.random_seed
            }
            models = [NHITS(**params)]

        elif self.model_type == 'TFT':
            params = {
                'h': self.horizon,
                'loss': self.loss,
                'max_steps': self.max_steps,
                'hist_exog_list': self.final_columns,
                'input_size': 2*self.horizon,
                'learning_rate': prms['learning_rate'],
                'hidden_size': prms['hidden_size'],
                'scaler_type': self.scaler_type,
                'random_seed': self.random_seed
            }
            models = [TFT(**params)]

        else:
            raise ValueError('Wrong model type!')
        
        params['loss'] = str(params['loss'])
        
            
        loss_val, predictions_val = pipeline_train_predict(models, self.train_set, self.val_set, self.horizon, self.loss_func, self.model_type)
        predictions_val = pd.concat([predictions_val, self.val_set_all[['Date']]], axis=1)
        
        train_val_set = pd.concat([self.train_set, self.val_set])
        loss_test, predictions_test = pipeline_train_predict(models, train_val_set, self.test_set, self.horizon, self.loss_func, self.model_type)
        predictions_test = pd.concat([predictions_test, self.test_set_all[['Date']]], axis=1)
        
        # create directiories if not exist
    
        if not os.path.exists(f'horizon_results/{self.prefix}'):
            os.mkdir(f'horizon_results/{self.prefix}')
        if not os.path.exists(f'horizon_results/{self.prefix}/{self.mode}'):
            os.mkdir(f'horizon_results/{self.prefix}/{self.mode}')
        if not os.path.exists(f'horizon_results/{self.prefix}/{self.mode}/{self.model_type}'):
            os.mkdir(f'horizon_results/{self.prefix}/{self.mode}/{self.model_type}')
        # All saves
        # Save params as json
        with open(f'horizon_results/{self.prefix}/{self.mode}/{self.model_type}/params_{self.timestamp}.json', 'w') as f:
            json.dump(params, f)
        # Save predictions_val and predictions_test as csv
        predictions_val.to_csv(f'horizon_results/{self.prefix}/{self.mode}/{self.model_type}/val_pred_{self.timestamp}.csv', index=False)
        predictions_test.to_csv(f'horizon_results/{self.prefix}/{self.mode}/{self.model_type}/test_pred_{self.timestamp}.csv', index=False)
        # Save loss_val and loss_test as json
        scores = {
            'val': loss_val,
            'test': loss_test
        }
        with open(f'horizon_results/{self.prefix}/{self.mode}/{self.model_type}/loss_{self.timestamp}.json', 'w') as f:
            json.dump(scores, f)

def main():
   
    # Example of usage
    prefix = 'NFLX' # Could be 'NFLX', 'BTC-USD', 'BA', 'TSLA'
    modes = ['financial'] # Could be 'financial', 'news', 'financial_and_news'
    model_types = ['NBEATSx', 'TFT'] # Could be 'NBEATSx', 'NHITS', 'TFT'
    max_steps = [20, 10]
    scaler_type = 'standard'# Better not to change
    loss=MSE()
    n_trials = 10
    random_seed = 1

    final_columns = [
        ['norm_rsi_14', 'norm_slowk_14', 'norm_rsi_gspc_14', 'minmax_daily_variation', f'minmax_{prefix}_Volume'],
        ['finbert_Score', 'vader_Score', 'mean_influential'],
        ['norm_rsi_14', 'norm_slowk_14', 'norm_rsi_gspc_14', 'minmax_daily_variation', f'minmax_{prefix}_Volume', 'finbert_Score', 'vader_Score', 'mean_influential']
    ] # Specify exogenous variables for each mode

    # Load data
    train_set_all = pd.read_csv('csv/'+prefix+'/train_set_full.csv')
    val_set_all = pd.read_csv('csv/'+prefix+'/val_set_full.csv')
    test_set_all = pd.read_csv('csv/'+prefix+'/test_set_full.csv')
    # Specify horizons and targets
    horizons = [5]
    targets = [f'{prefix}_Close', 'log_return_1', 'log_return_5']
    for target in targets:
        for horizon in horizons:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if target == f'{prefix}_Close':
                loss_func = mean_squared_error
            else:
                loss_func = eval_score
            for i, mode in enumerate(modes):
                for j, model_type in enumerate(model_types):
                    print(f'Prefix: {prefix} | mode: {mode} | model_type: {model_type}')
                    # Train models and save results
                    ht = HorizonTrainer(prefix, mode, model_type, final_columns[i], train_set_all, val_set_all, test_set_all, horizon, max_steps[j], scaler_type, loss_func, loss, n_trials, random_seed, timestamp, target)
                    ht.train()

if __name__ == "__main__":
    main()