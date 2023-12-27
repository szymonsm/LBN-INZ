import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
from neuralforecast.models import NBEATS, NHITS, TFT
from neuralforecast import NeuralForecast
from functools import partial

#FINAL_COLS = ['mean_future', 'mean_influential','mean_trustworthy', 'mean_clickbait','norm_rsi_14', 'norm_rsi_gspc_14', 'norm_slowk_14']
NAME_MODEL = {"NBEATS": NBEATS, "NHITS": NHITS, "TFT": TFT}

def create_trial_params_tft(trial):

    #input_size = trial.suggest_categorical('input_size', [1,2,3])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 12, 16, 20, 24, 28, 32])
    #dropout = trial.suggest_float('dropout', 0.0, 0.5)
    #attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.5)

    # model_params = {"learning_rate": learning_rate, "hidden_size": hidden_size, "dropout": dropout, "attn_dropout": attn_dropout}
    model_params = {"learning_rate": learning_rate, "hidden_size": hidden_size}

    return model_params


# TODO - żeby korzystać z NBEATS lub NHITS, trzeba uzupełnić odpowiednią funkcję, tak jak wyżej jest dla FEDformer
def create_trial_params_nbeats(trial):
    #input_size = trial.suggest_categorical('input_size', [1,2,3])
    
    n_blocks_season = trial.suggest_int('n_blocks_season', 1, 3)
    n_blocks_trend = trial.suggest_int('n_blocks_trend', 1, 3)
    n_blocks_identity = trial.suggest_int('n_blocks_ident', 1, 3)
    
    mlp_units_n = trial.suggest_categorical('mlp_units', [32, 64, 128])
    num_hidden = trial.suggest_int('num_hidden', 1, 3)
    
    n_harmonics = trial.suggest_int('n_harmonics', 1, 5)
    n_polynomials = trial.suggest_int('n_polynomials', 1, 5)
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
    
    n_blocks = [n_blocks_season, n_blocks_trend, n_blocks_identity]
    #n_blocks = [n_blocks_trend, n_blocks_identity]
    mlp_units=[[mlp_units_n, mlp_units_n]]*num_hidden
    
    model_params = {
      "stack_types": ['seasonality', 'trend', 'identity'], "mlp_units": mlp_units, "n_blocks": n_blocks,
      "n_harmonics": n_harmonics, "n_polynomials": n_polynomials, "learning_rate": learning_rate}
    
    return model_params

def create_trial_params_nhits(trial):
    #input_size = trial.suggest_categorical('input_size', [1,2,3])
    
    n_blocks1 = trial.suggest_int('n_blocks1', 1, 3)
    n_blocks2 = trial.suggest_int('n_blocks2', 1, 3)
    n_blocks3 = trial.suggest_int('n_blocks3', 1, 3)
    
    mlp_units_n = trial.suggest_categorical('mlp_units', [32, 64, 128])
    
    n_pool_kernel_size1 = trial.suggest_int('n_pool_kernel_size1', 1, 3)
    n_pool_kernel_size2 = trial.suggest_int('n_pool_kernel_size2', 1, 3)
    n_pool_kernel_size3 = trial.suggest_int('n_pool_kernel_size3', 1, 3)

    #n_freq_downsample1 = trial.suggest_int('n_freq_downsample1', 1, 5)
    #n_freq_downsample2 = trial.suggest_int('n_freq_downsample2', 1, 5)
    #n_freq_downsample3 = trial.suggest_int('n_freq_downsample3', 1, 5)
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    #dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.0, 0.5)
    
    
    n_blocks = [n_blocks1, n_blocks2, n_blocks3]
    mlp_units=[[mlp_units_n, mlp_units_n]]*3
    n_pool_kernel_size = [n_pool_kernel_size1, n_pool_kernel_size2, n_pool_kernel_size3]
    #n_freq_downsample = [n_freq_downsample1, n_freq_downsample2, n_freq_downsample3]
    stack_types = ['identity', 'identity', 'identity']
    
    # model_params = {"mlp_units": mlp_units, "n_blocks": n_blocks, "stack_types": stack_types,
    # "learning_rate": learning_rate, "dropout_prob_theta": dropout_prob_theta, "n_pool_kernel_size": n_pool_kernel_size,
    # "n_freq_downsample": n_freq_downsample}
    model_params = {"mlp_units": mlp_units, "n_blocks": n_blocks, "stack_types": stack_types,
    "learning_rate": learning_rate, "n_pool_kernel_size": n_pool_kernel_size}
    
    return model_params



def pipeline_train_predict(models, train_set, val_set, horizon, loss_func, model_name):
    #losses = []
    predictions = None
    val_set_begin = val_set.iloc[0]['ds']

    train_set_tmp = train_set
    last_index = horizon
    val_set_tmp = val_set.iloc[:last_index]
    modelID = 0
    while train_set_tmp.iloc[0]['ds'] < val_set_begin and len(val_set_tmp) == horizon:
        #print(len(train_set_tmp), len(val_set_tmp))
        #val_set_tmp = val_set_tmp[val_set_tmp['y'].notnull()]
        predictions_tmp = train(models, train_set_tmp, val_set_tmp, model_name)
        predictions_tmp['modelID'] = modelID
        modelID += 1
        #losses.append(loss)
        predictions = predictions.append(predictions_tmp) if predictions is not None else predictions_tmp

        # Update sets
        # append first horizon rows to train_set_tmp from val_set_tmp
        train_set_tmp = train_set_tmp.append(val_set_tmp)
        # make val_set_tmp start from the next day after the last day of train_set_tmp
        val_set_tmp = val_set.iloc[last_index:last_index+horizon]
        last_index = last_index + horizon
    # Add sequence ID and reset index
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
    # remove rows where y is null
    # p = p[p['y'].notnull()]
    # print(p['y'])
    # print(p[f'{model_name}-median'])
    # print(len(p['y']), len(p[f'{model_name}-median']))
    # print(metric_function(p['y'], p[f'{model_name}-median']))
    # return metric_function(p['y'], p[f'{model_name}-median']), p
    return p

def create_trial_params(trial, model_name: str):
    if model_name == "TFT":
      return create_trial_params_tft(trial)
    if model_name == "NBEATS":
      return create_trial_params_nbeats(trial)
    if model_name == "NHITS":
      return create_trial_params_nhits(trial)     


def objective(trial, train_set, val_set, loss, model_name: str, horizon: int, hist_exog_list: list, max_steps: int, random_seed: int, loss_func: callable, scaler_type = 'standard'):
    model_params = create_trial_params(trial, model_name)
    # n_blocks = [prms['n_blocks_season'], prms['n_blocks_trend'], prms['n_blocks_ident']]
    # mlp_units=[[prms['mlp_units'], prms['mlp_units']]*prms['num_hidden']]
    models = [NAME_MODEL[model_name](h=horizon, input_size=2*horizon, loss=loss, hist_exog_list=hist_exog_list, random_seed=random_seed, scaler_type=scaler_type, max_steps=max_steps, **model_params)]

    loss, predictions = pipeline_train_predict(models, train_set, val_set, horizon, loss_func, model_name)
    # select rows where predictions['sequenceID'] == horizon-1
    # p = predictions[predictions['sequenceID'] == horizon-1]
    # l = loss_func(p['y'], p[f'{model_name}-median'])
    # return l
    return loss


def main():
   
    # Przykłady użycia

    # Tunowanie hiperparametrów
    study = optuna.create_study(direction='minimize')
    study.optimize(partial(objective, train_set=train_set, val_set=val_set, 
                    model_name="FEDformer", horizon=14,loss_func=mean_squared_error), n_trials=5)
    
    # Predykcja na podstawie najlepszych hiperparametrów
    horizon = 14
    models = [FEDformer(
                    h=horizon,
                    #loss=DistributionLoss(distribution='Normal', level=[90]),
                    max_steps=100,
                    futr_exog_list=FINAL_COLS,
                    input_size=study.best_params['input_size'],
                    scaler_type=study.best_params['scaler_type'],
                    learning_rate=study.best_params['learning_rate'],
                    random_seed=1
                    )]
    loss, predictions = train(models, train_set, val_set, mean_squared_error, "FEDformer")
    print("MSE: ", loss)

    # Pipeline, który robi predykcję batchami co ileś dni, ustalane przez horizon
    loss, predictions = pipeline_train_predict(models, train_set, val_set, horizon, mean_squared_error, "FEDformer")

if __name__ == "__main__":
    main()