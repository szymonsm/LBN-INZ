'''
Funkcje pomocnicze do analizy wyników modeli horizonowych.
'''
from CODE.model_training.scripts.horizon_model_training import *
import pandas as pd
import numpy as np
import json
from itertools import product 
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import matplotlib.pyplot as plt
import optuna

def combo_plot(prefix, mode, timestamps, timestamps_NBEATSx, _type='val'):
    """
    Plot predictions for all models for a given prefix, mode and type of data (val or test).

    Parameters:
    - prefix (str): The prefix of the dataset.
    - mode (str): The mode of the dataset (financial, news, financial_and_news).
    - timestamps (dict): Dictionary containing timestamps for all models for a given prefix and mode.
    - timestamps_NBEATSx (dict): Dictionary containing timestamps for NBEATSx for a given prefix and mode.
    - _type (str): The type of data (val or test).

    Returns:
    - fig (matplotlib.pyplot.figure): The figure containing all plots.
    """
    j=1
    fig = plt.figure(figsize=(10, 10))
    if prefix == 'BTC-USD':
        second_horizon = 7
    else:
        second_horizon = 5

    for i in range(3):
        for model_type in ['NBEATSx', 'NHITS', 'TFT']:
            if model_type != 'NBEATSx':
                if prefix == 'BTC-USD' and mode == 'news' and model_type == 'NBEATS':
                    pred2 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[0][i*2]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[0][i*2]}.json') as json_file:
                        loss2 = json.load(json_file)[f'{_type}']
                    pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[0][i*2+1]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[0][i*2+1]}.json') as json_file:
                        loss5 = json.load(json_file)[f'{_type}']
                elif prefix == 'BTC-USD' and mode == 'news' and (model_type == 'NHITS' or model_type == 'TFT'):
                    pred2 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[1][i*2]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[1][i*2]}.json') as json_file:
                        loss2 = json.load(json_file)[f'{_type}']
                    pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[1][i*2+1]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[1][i*2+1]}.json') as json_file:
                        loss5 = json.load(json_file)[f'{_type}']
                else:
                    pred2 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[i*2]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[i*2]}.json') as json_file:
                        loss2 = json.load(json_file)[f'{_type}']
                    pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[i*2+1]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[i*2+1]}.json') as json_file:
                        loss5 = json.load(json_file)[f'{_type}']
            else:
                pred2 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps_NBEATSx[i*2]}.csv')
                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps_NBEATSx[i*2]}.json') as json_file:
                    loss2 = json.load(json_file)[f'{_type}']
                pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps_NBEATSx[i*2+1]}.csv')
                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps_NBEATSx[i*2+1]}.json') as json_file:
                    loss5 = json.load(json_file)[f'{_type}']

            ax1 = fig.add_subplot(3, 3, j)
            ax1.plot(pred2['y'].reset_index(drop=True), label='actual')
            ax1.plot(pred2[f'{model_type}'].reset_index(drop=True), label=f'horizon: 2')
            ax1.plot(pred5[f'{model_type}'].reset_index(drop=True), label=f'horizon: {second_horizon}')
            ax1.annotate(f'h=2, loss: {loss2:.2f}', xy=(0.05, 0.93), xycoords='axes fraction')
            ax1.annotate(f'h={second_horizon}, loss: {loss5:.2f}', xy=(0.05, 0.87), xycoords='axes fraction')
            if j==2:
                ax1.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncols=3)
            if j in [1,2,3]:
                ax1.set_title(f'{model_type}', size='large')
            if j in [1,4,7]:
                if j == 1:
                    ax1.set_ylabel(f'{prefix}_Close', size='large')
                elif j == 4:
                    ax1.set_ylabel(f'log_return_1', size='large')
                else:
                    ax1.set_ylabel(f'log_return_{second_horizon}', size='large')
            if j in [7,8,9]:
                ax1.set_xlabel(f'Day', size='large')
            j+=1
    return fig

def combo_plot_for_prefix(prefix, TIMESTAMPS, TIMESTAMPS_NBEATSx, _type='val'):
    """
    Plot predictions for all models for a given prefix, mode and type of data (val or test).

    Parameters:
    - prefix (str): The prefix of the dataset.
    - mode (str): The mode of the dataset (financial, news, financial_and_news).
    - timestamps (dict): Dictionary containing timestamps for all models for a given prefix and mode.
    - timestamps_NBEATSx (dict): Dictionary containing timestamps for NBEATSx for a given prefix and mode.
    - _type (str): The type of data (val or test).

    Returns:
    - fig (matplotlib.pyplot.figure): The figure containing all plots.
    """
    modes = ['financial', 'news', 'financial_and_news']
    j=1
    fig = plt.figure(figsize=(10, 10))
    if prefix == 'BTC-USD':
        second_horizon = 7
    else:
        second_horizon = 5

    for mode in modes:
        timestamps = TIMESTAMPS[prefix][mode]
        timestamps_NBEATSx = TIMESTAMPS_NBEATSx[prefix][mode]
        for model_type in ['NBEATSx', 'NHITS', 'TFT']:
            i = 2
            if model_type != 'NBEATSx':
                if prefix == 'BTC-USD' and mode == 'news' and model_type == 'NBEATS':
                    pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[0][i*2+1]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[0][i*2+1]}.json') as json_file:
                        loss5 = json.load(json_file)[f'{_type}']
                elif prefix == 'BTC-USD' and mode == 'news' and (model_type == 'NHITS' or model_type == 'TFT'):
                    pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[1][i*2+1]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[1][i*2+1]}.json') as json_file:
                        loss5 = json.load(json_file)[f'{_type}']
                else:
                    pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps[i*2+1]}.csv')
                    with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps[i*2+1]}.json') as json_file:
                        loss5 = json.load(json_file)[f'{_type}']
            else:
                pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps_NBEATSx[i*2+1]}.csv')
                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps_NBEATSx[i*2+1]}.json') as json_file:
                    loss5 = json.load(json_file)[f'{_type}']

            ax1 = fig.add_subplot(3, 3, j)
            ax1.plot(pred5['y'].reset_index(drop=True), label='actual')
            ax1.plot(pred5[f'{model_type}'].reset_index(drop=True), label=f'horizon: {second_horizon}')
            ax1.scatter(pred5[pred5['sequenceID']==pred5['sequenceID'].max()].index, pred5[pred5['sequenceID']==pred5['sequenceID'].max()][f'{model_type}'], color='red', s=10, label='prediction day')
            ax1.annotate(f'loss: {loss5:.2f}', xy=(0.05, 0.93), xycoords='axes fraction')
            if j==2:
                ax1.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncols=3)
            if j in [1,2,3]:
                ax1.set_title(f'{model_type}', size='large')
            if j in [1,4,7]:
                if j == 1:
                    ax1.set_ylabel(f'financial', size='large')
                elif j == 4:
                    ax1.set_ylabel(f'news', size='large')
                else:
                    ax1.set_ylabel(f'financial_and_news', size='large')
            if j in [7,8,9]:
                ax1.set_xlabel(f'Day', size='large')
            j+=1
    fig.suptitle(f'{prefix} | {_type}', size='xx-large')
    return fig

def eval_score(y_true, y_pred):
    """
    Calculate evaluation score for time series predictions.

    Parameters:
    - y_true, y_pred: True and predicted values.

    Returns:
    - float: Evaluation score.
    """
    csm = count_signs_matrix(y_true.reset_index(drop=True), y_pred.reset_index(drop=True))
    return (3*csm[1][1]/(csm[1][0]+csm[1][1]+0.0001) + 2*csm[1][1]/(csm[0][1]+csm[1][1]+0.0001))/5

def calculate_loss_baseline_prev(df, target):
    """
    Calculate baseline loss for a given target using previous value.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the predictions.
    - target (str): The target column.
    
    Returns:
    - float: The baseline loss.
    """
    if 'Close' in target:
        loss = mean_squared_error(df['y'][1:].reset_index(drop=True), df['y'].shift(1)[1:].reset_index(drop=True))
    else:
        loss = f1_score(df['y'][1:], df['y'].shift(1)[1:])
    return loss

def calculate_loss_baseline_horizon(df, target, horizon):
    """
    Calculate baseline loss for a given target using horizon value.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the predictions.
    - target (str): The target column.
    - horizon (int): The horizon.
    
    Returns:
    - float: The baseline loss.
    """

    y_true = df['y'][horizon:].reset_index(drop=True)
    i = horizon-1
    preds = []
    while i < len(df['y']) - 1:
        preds += [df['y'][i]] * horizon
        i += horizon
    y_pred = pd.Series(preds)
    
    if 'Close' in target:
        loss = mean_squared_error(y_true.reset_index(drop=True), y_pred.reset_index(drop=True))
    else:
        loss = eval_score(y_true, y_pred)
    return loss

def get_close_pred(df_prediction, df, prefix, step, model_name):   
    df_pred = df_prediction.copy() 
    row_number_start = df.loc[df['Date'] == df_pred.iloc[0]['Date']].index[0]
    row_number_end = df.loc[df['Date'] == df_pred.iloc[-1]['Date']].index[0]
    df_pred['Close_pred'] = np.exp(df_pred[model_name]) *  df[prefix+'_Close'][row_number_start-step:row_number_end+1-step].values.flatten()
    df_pred['Close'] = df[prefix+'_Close'][row_number_start:row_number_end+1].values.flatten()
    df_pred['Open'] = df[prefix+'_Open'][row_number_start:row_number_end+1].values.flatten()
    
    open = df_pred['Open'].values.flatten()[0::step]
    df_pred['Price_buy'] = [element for element in open for _ in range(step)]

    df_pred['Ratio_pred'] = ((df_pred['Close_pred'] - df_pred['Price_buy']) / df_pred['Price_buy'])*100

    return df_pred

def calculate_return(df, intrest, thr_p, thr_m):
    return_list = []
    for _, row in df.iterrows():
        #if row['sequenceID'] < idx_trust:
        if row['Ratio_pred'] > thr_p:
            money_prc = (row['Close']*(1-intrest) - row['Price_buy']*(1+intrest))/row['Price_buy']
            return_list.append(money_prc)

        if row['Ratio_pred'] < thr_m:
            money_prc = (row['Price_buy']*(1-intrest) - row['Close']*(1+intrest))/row['Price_buy']
            return_list.append(money_prc)

    if return_list:
        return np.mean(return_list), return_list
    else:
        return -0.50, None

def objective(trial, df, intrest):
    thr_p = trial.suggest_uniform('thr_p', 0.0001, 20)
    thr_m = trial.suggest_uniform('thr_m', -20, -0.0001)
    # idx_trust = trial.suggest_int('idx_trust', 0, horizon)

    mean_return, _ = calculate_return(df, intrest, thr_p, thr_m)
    return mean_return

def optimize_parameters_with_optuna(df, intrest):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df, intrest), n_trials=600) 

    best_params = study.best_params
    max_mean_return = study.best_value  

    return best_params, max_mean_return

def create_summary_df(timestamps, timestamps_NBEATSx):
    """
    Create a summary DataFrame containing all results for all models.
    
    Parameters:
    - timestamps (dict): Dictionary containing timestamps for all models for a given prefix and mode.
    - timestamps_NBEATSx (dict): Dictionary containing timestamps for NBEATSx for a given prefix and mode.
    
    Returns:
    - df_summary (pd.DataFrame): DataFrame containing all results for all models.
    """
    df_summary = pd.DataFrame(columns=['prefix', 'target', 'model', 'mode', 'loss','Baseline-prev', 'Baseline-horizon', 'horizon', 'return_no_interest', 'return_interest'])

    for prefix in timestamps:
        if prefix == 'BTC-USD':
            second_horizon = 7
            df = pd.read_csv(f'csv/{prefix}/{prefix}_with_weekends.csv')
        else:
            second_horizon = 5
            df = pd.read_csv(f'csv/{prefix}/{prefix}_without_weekends.csv')
        for i in range(3): # targets
            row_2 = {}
            row_5 = {}
            row_2['prefix'] = prefix
            row_5['prefix'] = prefix
            if i == 0:
                row_2['target'] = 'Close'
                row_5['target'] = 'Close'
            elif i == 1:
                row_2['target'] = 'log_return_1'
                row_5['target'] = 'log_return_1'
            else:
                row_2['target'] = 'log_return_'+str(second_horizon)
                row_5['target'] = 'log_return_'+str(second_horizon)
            best_params5_interest_dict = dict().fromkeys(['NBEATSx-financial', 'NBEATSx-news', 'NBEATSx-financial_and_news', 'NHITS-financial', 'NHITS-news', 'NHITS-financial_and_news', 'TFT-financial', 'TFT-news', 'TFT-financial_and_news'])
            best_params5_dict = dict().fromkeys(['NBEATSx-financial', 'NBEATSx-news', 'NBEATSx-financial_and_news', 'NHITS-financial', 'NHITS-news', 'NHITS-financial_and_news', 'TFT-financial', 'TFT-news', 'TFT-financial_and_news'])
            for _type in ['val', 'test']:
                for mode in timestamps[prefix]:
                    ts = timestamps[prefix][mode]
                    row_2['mode'] = mode
                    row_5['mode'] = mode
                    for model_type in ['NBEATSx', 'NHITS', 'TFT']:
                        row_2['model'] = model_type
                        row_5['model'] = model_type
                        if model_type != 'NBEATSx':
                            if prefix == 'BTC-USD' and mode == 'news' and model_type == 'NBEATS':
                                pred2 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{ts[0][i*2]}.csv')
                                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{ts[0][i*2]}.json') as json_file:
                                    loss2 = json.load(json_file)[f'{_type}']
                                pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{ts[0][i*2+1]}.csv')
                                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{ts[0][i*2+1]}.json') as json_file:
                                    loss5 = json.load(json_file)[f'{_type}']
                            elif prefix == 'BTC-USD' and mode == 'news' and (model_type == 'NHITS' or model_type == 'TFT'):
                                pred2 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{ts[1][i*2]}.csv')
                                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{ts[1][i*2]}.json') as json_file:
                                    loss2 = json.load(json_file)[f'{_type}']
                                pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{ts[1][i*2+1]}.csv')
                                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{ts[1][i*2+1]}.json') as json_file:
                                    loss5 = json.load(json_file)[f'{_type}']
                            else:
                                pred2 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{ts[i*2]}.csv')
                                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{ts[i*2]}.json') as json_file:
                                    loss2 = json.load(json_file)[f'{_type}']
                                pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{ts[i*2+1]}.csv')
                                with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{ts[i*2+1]}.json') as json_file:
                                    loss5 = json.load(json_file)[f'{_type}']
                            row_2['loss'] = round(loss2,2)
                            row_5['loss'] = round(loss5,2)
                        else:
                            pred2 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps_NBEATSx[prefix][mode][i*2]}.csv')
                            with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps_NBEATSx[prefix][mode][i*2]}.json') as json_file:
                                loss2 = json.load(json_file)[f'{_type}']
                            pred5 = pd.read_csv(f'horizon_results/{prefix}/{mode}/{model_type}/{_type}_pred_{timestamps_NBEATSx[prefix][mode][i*2+1]}.csv')
                            with open(f'horizon_results/{prefix}/{mode}/{model_type}/loss_{timestamps_NBEATSx[prefix][mode][i*2+1]}.json') as json_file:
                                loss5 = json.load(json_file)[f'{_type}']
                            row_2['loss'] = round(loss2,2)
                            row_5['loss'] = round(loss5,2)
                        if row_5['target'] != 'log_return_'+str(second_horizon):
                            row_2['return_interest'] = None
                            row_5['return_interest'] = None
                            row_2['return_no_interest'] = None
                            row_5['return_no_interest'] = None
                        else:
                            row_2['return_interest'] = None
                            row_2['return_no_interest'] = None
                            if _type == 'val':
                                pred5 = get_close_pred(pred5, df, prefix, second_horizon, model_type)
                                best_params5_interest, max_mean_return5_interest = optimize_parameters_with_optuna(pred5, second_horizon, rate=0.002)
                                best_params5, max_mean_return5 = optimize_parameters_with_optuna(pred5, second_horizon, rate=0)
                                best_params5_interest_dict[f'{model_type}-{mode}'] = best_params5_interest
                                best_params5_dict[f'{model_type}-{mode}'] = best_params5
                                row_5['return_interest'] = max_mean_return5_interest
                                row_5['return_no_interest'] = max_mean_return5
                            else:
                                pred5 = get_close_pred(pred5, df, prefix, second_horizon, model_type)
                                row_5['return_interest'] = calculate_return(pred5, *best_params5_interest_dict[f'{model_type}-{mode}'])[0]
                                row_5['return_no_interest'] = calculate_return(pred5, *best_params5_dict[f'{model_type}-{mode}'])[0]
                        row_2['set'] = _type
                        row_5['set'] = _type
                        # calculate baseline losses
                        if _type == 'val':
                            row_2['Baseline-prev'] = round(calculate_loss_baseline_prev(pred2, row_2['target']),2)
                            row_5['Baseline-prev'] = round(calculate_loss_baseline_prev(pred5, row_5['target']),2)
                            row_2['Baseline-horizon'] = round(calculate_loss_baseline_horizon(pred2, row_2['target'], 2),2)
                            row_5['Baseline-horizon'] = round(calculate_loss_baseline_horizon(pred5, row_5['target'], second_horizon),2)
                        else:
                            row_2['Baseline-prev'] = round(calculate_loss_baseline_prev(pred2, row_2['target']),2)
                            row_5['Baseline-prev'] = round(calculate_loss_baseline_prev(pred5, row_5['target']),2)
                            row_2['Baseline-horizon'] = round(calculate_loss_baseline_horizon(pred2, row_2['target'], 1),2)
                            row_5['Baseline-horizon'] = round(calculate_loss_baseline_horizon(pred5, row_5['target'], second_horizon),2)
                        row_2['horizon'] = 2
                        row_5['horizon'] = second_horizon
                        df_summary = df_summary.append(row_2, ignore_index=True)
                        df_summary = df_summary.append(row_5, ignore_index=True)
    return df_summary

def get_best_model(summary, prefix, _type):
    """
    Get the best model for a given prefix and type of data (val or test).
    
    Parameters:
    - summary (pd.DataFrame): DataFrame containing all results for all models.
    - prefix (str): The prefix of the dataset.
    - _type (str): The type of data (val or test).
    
    Returns:
    - str: The best model.
    - str: The mode of the best model.
    - str: The target of the best model.
    """
    df = summary[(summary['prefix'] == prefix) & (summary['set'] == _type) & (summary['target'] == 'log_return_5')]
    best_model = df[df['loss'] == df['loss'].max()]
    return best_model['model'].values[0], best_model['mode'].values[0], best_model['target'].values[0]

def concat_is_better_than_baseline(summary):
    """
    Concatenate the 'is_better' column to the summary DataFrame.
    
    Parameters:
    - summary (pd.DataFrame): DataFrame containing all results for all models.
    
    Returns:
    - pd.DataFrame: DataFrame containing all results for all models.
    """

    is_better = []
    for _, row in summary.iterrows():
        if row['target'] == 'Close':
            if row['loss'] < row['Baseline-horizon']:
                is_better += [1]
            else:
                is_better += [0]
        else:
            if row['loss'] > row['Baseline-horizon']:
                is_better += [1]
            else:
                is_better += [0]
    summary['is_better'] = is_better
    return summary

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate mean absolute percentage error (MAPE).

    Parameters:
    - y_true, y_pred: True and predicted values.

    Returns:
    - float: MAPE.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_metrics(y_true, y_pred):
    """
    Calculate common metrics for time series predictions.

    Parameters:
    - y_true_train, y_pred_train: True and predicted values for the training set.

    Returns:
    - pd.DataFrame: DataFrame with calculated metrics for each set (train, test, val).
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
    Calculate common classification metrics for time series predictions.

    Parameters:
    - y_true, y_pred: True and predicted values.

    Returns:
    - pd.DataFrame: DataFrame with calculated metrics for each set (train, test, val).
    """

    y_pred_probabilities = expit(y_pred)
    y_true_binary = (y_true > 0).astype(int) 
    y_pred_binary = (y_pred > 0).astype(int) 

    return {
        'Accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'Precision': precision_score(y_true_binary, y_pred_binary),
        'Recall': recall_score(y_true_binary, y_pred_binary),
        'F1 Score': f1_score(y_true_binary, y_pred_binary),
        'AUC': roc_auc_score(y_true_binary, y_pred_probabilities)
    }

def baseline_horizon_plot(df, target, horizon):
    """
    Plot baseline predictions for a given target using horizon value.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the predictions.
    - target (str): The target column.
    - horizon (int): The horizon.

    Returns:
    - fig (matplotlib.pyplot.figure): The figure containing all plots.
    """

    y_true = df['y'][horizon:].reset_index(drop=True)
    i = horizon-1
    preds = []
    while i < len(df['y']) - 1:
        preds += [df['y'][i]] * horizon
        i += horizon
    preds_ = [None] * (horizon) + preds 
    y_pred_ = pd.Series(preds_)   
    y_pred = pd.Series(preds)

    loss = eval_score(y_true, y_pred)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    # date as index
    ax1.plot(df['y'].reset_index(drop=True), label='actual')
    ax1.plot(y_pred_, label=f'horizon: {horizon}')

    df_cut = df[df['Date'] >= df['Date'][horizon-1]]
    
    ax1.scatter(df_cut[df_cut['sequenceID']==df_cut['sequenceID'].max()].index, df_cut[df_cut['sequenceID']==df_cut['sequenceID'].max()]['y'], color='red', s=10, label='prediction day')
    ax1.annotate(f'loss: {loss:.2f}', xy=(0.05, 0.93), xycoords='axes fraction')
    ax1.set_ylabel(f'{target}')
    ax1.set_xlabel(f'Day')
    ax1.legend()
    return fig
