from model_training import *
import pandas as pd
import numpy as np
from scripts.essentials import *
from scripts.plots import *
from scripts.train_utilities import *
from neuralforecast.losses.pytorch import MQLoss, MSE
import json
import os
import copy
from sklearn.metrics import mean_squared_error, mean_squared_log_error

class HorizonTrainer:
    def __init__(self, prefix, mode, model_type, final_columns, train_set_all, val_set_all, test_set_all, horizon, max_steps, scaler_type, loss_func, loss, random_seed):
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
        self.random_seed = random_seed
    
    def prepare_single_set(self, input_set):
        set_ = input_set.copy()
        set_ = set_[['DateGen', f'{self.prefix}_Close'] + self.final_columns]
        set_.rename(columns={'DateGen':'ds', f'{self.prefix}_Close':'y'}, inplace=True) 
        set_['unique_id'] = self.prefix 
        return set_

    def __prepare_sets__(self):

        self.train_set_all['DateGen'] = pd.date_range(start='2021-01-01', end='2023-11-26', freq='D')[:len(self.train_set)]
        valid_start_date = self.train_set['DateGen'].max() + pd.DateOffset(days=1)
        self.val_set_all['DateGen'] = pd.date_range(start=valid_start_date, end='2023-11-26', freq='D')[:len(self.val_set)]
        test_start_date = self.val_set['DateGen'].max() + pd.DateOffset(days=1)
        self.test_set_all['DateGen'] = pd.date_range(start=test_start_date, end='2023-11-26', freq='D')[:len(self.test_set)]

        self.train_set = self.prepare_single_set(self.train_set_all)
        self.val_set = self.prepare_single_set(self.val_set_all)
        self.test_set = self.prepare_single_set(self.test_set_all)

    def tune_params(self, objective_func: callable, n_trials: int):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_func, n_trials=n_trials)
        
        return study.best_trial.params
    
    def save_to_dir(self, params, predictions_val, predictions_test, loss_val, loss_test):
        if not os.path.exists(f'D:/pw/Thesis/LBN-INZ/CODE/model_training/results/{self.prefix}'):
            os.makedirs(f'D:/pw/Thesis/LBN-INZ/CODE/model_training/results/{self.prefix}/{self.mode}/{self.model_type}')
        import datetime
        with open(f'results/{self.prefix}/{self.mode}/{self.model_type}/params_{datetime.datetime.now()}.json', 'w') as f:
            json.dump(params, f)

        # Save predictions_val and predictions_test as csv
        predictions_val.to_csv(f'D:/pw/Thesis/LBN-INZ/CODE/model_training/results/{self.prefix}/{self.mode}/{self.model_type}/val_pred.csv', index=False)
        predictions_test.to_csv(f'D:/pw/Thesis/LBN-INZ/CODE/model_training/results/{self.prefix}/{self.mode}/{self.model_type}/test_pred.csv', index=False)
        # Save loss_val and loss_test as json
        scores = {
            'val': loss_val,
            'test': loss_test
        }
        with open(f'results/{self.prefix}/{self.mode}/{self.model_type}/loss.json', 'w') as f:
            json.dump(scores, f)

    def update_tuned_params(self, basic_params, model_tuned_params):
        if self.model_type == 'NBEATS':
            n_blocks = [model_tuned_params['n_blocks_season'], model_tuned_params['n_blocks_trend'],
                         model_tuned_params['n_blocks_ident']]
            mlp_units=[[model_tuned_params['mlp_units'], model_tuned_params['mlp_units']]]*3, 
            model_tuned_params['num_hidden']
            basic_params.update({
                'stack_types': ['seasonality', 'trend', 'identity'],
                'mlp_units': mlp_units,
                'n_blocks': n_blocks,
                'learning_rate': model_tuned_params['learning_rate'],
                'n_harmonics': model_tuned_params['n_harmonics'],
                'n_polynomials': model_tuned_params['n_polynomials'],
            })
            models = [NBEATS(**basic_params)]

        elif self.model_type == 'NHITS':
            n_blocks = [model_tuned_params['n_blocks1'], model_tuned_params['n_blocks2'], model_tuned_params['n_blocks3']]
            mlp_units=[[model_tuned_params['mlp_units'], model_tuned_params['mlp_units']]]*3
            n_pool_kernel_size = [model_tuned_params['n_pool_kernel_size1'], model_tuned_params['n_pool_kernel_size2'], model_tuned_params['n_pool_kernel_size3']]
            n_freq_downsample = [model_tuned_params['n_freq_downsample1'], model_tuned_params['n_freq_downsample2'], model_tuned_params['n_freq_downsample3']]
            basic_params.update({
                'stack_types': ['identity', 'identity', 'identity'],
                'mlp_units': mlp_units,
                'n_blocks': n_blocks,
                'learning_rate': model_tuned_params['learning_rate'],
                'n_pool_kernel_size': n_pool_kernel_size,
                'n_freq_downsample': n_freq_downsample,
                'dropout_prob_theta': model_tuned_params['dropout_prob_theta'],
            })
            models = [NHITS(**basic_params)]

        elif self.model_type == 'TFT':
            basic_params.update({
                'learning_rate': model_tuned_params['learning_rate'],
                'hidden_size': model_tuned_params['hidden_size'],
                'dropout': model_tuned_params['dropout'],
                'attn_dropout': model_tuned_params['attn_dropout'],
            })
            models = [TFT(**basic_params)]

        else:
            raise ValueError('Wrong model type!')
        
        return models


    def train(self, objective_func: callable = objective, n_trials: int = 5):
        self.__prepare_sets__()

        val_set = self.val_set.iloc[:5, :]
        test_set = self.test_set.iloc[:5, :]

        params = {'h': self.horizon,
                'loss': self.loss,
                'max_steps': self.max_steps,
                'hist_exog_list': self.final_columns,
                'input_size': 2*self.horizon,
                'scaler_type': self.scaler_type,
                'random_seed': self.random_seed}
        prms = self.tune_params(partial(objective_func, train_set=self.train_set, val_set=val_set,
                                model_name=self.model_type, loss_func=self.loss_func, **params),
                                n_trials=n_trials)

        models = self.update_tuned_params(params, prms)
        params['loss'] = str(params['loss'])
        
        loss_val, predictions_val = pipeline_train_predict(models, self.train_set, val_set, self.horizon, self.loss_func, self.model_type)
        predictions_val = pd.concat([predictions_val, self.val_set_all[['Date']]], axis=1)
        
        train_val_set = pd.concat([self.train_set, self.val_set])
        loss_test, predictions_test = pipeline_train_predict(models, train_val_set, test_set, self.horizon, self.loss_func, self.model_type)
        predictions_test = pd.concat([predictions_test, self.test_set_all[['Date']].iloc[:5, :]], axis=1)
        
        self.save_to_dir(params, predictions_val, predictions_test, loss_val, loss_test)

def main():

    prefix = "NFLX"
    # modes = ['financial', 'news', 'financial_and_news']
    # model_types = ['NBEATS', 'NHITS', 'TFT']
    # final_columns = [
    #     ['norm_rsi_gspc_14', 'norm_rsi_14', 'norm_slowk_14', 'minmax_high_norm', 'log_return_1'],
    #     ['finbert_Score', 'bart_Score', 'vader_Score', 'mean_influential', 'mean_trustworthy'],
    #     ['finbert_Score', 'bart_Score', 'vader_Score', 'mean_influential', 'mean_trustworthy', 'norm_rsi_gspc_14', 'norm_rsi_14', 'norm_slowk_14', 'minmax_high_norm', 'log_return_1']
    # ]
    modes = ['financial_and_news']
    model_types = ['TFT']
    final_columns = [
        # ['norm_rsi_gspc_14', 'norm_rsi_14', 'norm_slowk_14', 'minmax_high_norm', 'log_return_1'],
        # ['finbert_Score', 'bart_Score', 'vader_Score', 'mean_influential', 'mean_trustworthy'],
        #['finbert_Score', 'bart_Score', 'vader_Score', 'mean_influential', 'mean_trustworthy', 'norm_rsi_gspc_14', 'norm_rsi_14', 'norm_slowk_14', 'minmax_high_norm', 'log_return_1'],
        ['finbert_Score', 'mean_influential', 'norm_rsi_gspc_14', 'minmax_high_norm', 'log_return_1']
        # ['finbert_Score', 'bart_Score', 'vader_Score', 'future_finbert',
        # 'future_bart', 'future_vader', 'influential_finbert',
        # 'influential_bart', 'influential_vader', 'trustworthy_finbert',
        # 'trustworthy_bart', 'trustworthy_vader', 'clickbait_finbert',
        # 'clickbait_bart', 'clickbait_vader','norm_rsi_14', 'norm_rsi_gspc_14', 'norm_slowk_14', 'vwap_14',
        # 'norm_roc_14', 'log_return_1', 'minmax_^GSPC_Volume', 'minmax_NFLX_Volume',
        # 'minmax_daily_variation', 'minmax_high_close_pressure',
        # 'minmax_low_open_pressure', 'minmax_low_norm', 'minmax_close_norm',
        # 'minmax_high_norm', 'minmax_open_norm' ]
    ]
    train_set_all = pd.read_csv('D:/pw/Thesis/LBN-INZ/CODE/model_training/csv/'+prefix+'/train_set_full.csv')
    val_set_all = pd.read_csv('D:/pw/Thesis/LBN-INZ/CODE/model_training/csv/'+prefix+'/val_set_full.csv')
    test_set_all = pd.read_csv('D:/pw/Thesis/LBN-INZ/CODE/model_training/csv/'+prefix+'/test_set_full.csv')
    horizon = 5
    val_set_all = val_set_all.iloc[:horizon, :]
    max_steps = [50]
    scaler_type = 'standard'
    loss_func = mean_squared_error
    loss=MSE()
    n_trials = 25
    random_seed = 1

    for i, mode in enumerate(modes):
        for j, model_type in enumerate(model_types):
            print(f'Prefix: {prefix} | mode: {mode} | model_type: {model_type}')
            ht = HorizonTrainer(prefix, mode, model_type, final_columns[i], train_set_all, val_set_all, test_set_all, horizon, max_steps[j], scaler_type, loss_func, loss, random_seed)
            ht.train(objective_func=objective, n_trials=n_trials)

if __name__ == '__main__':
    main()