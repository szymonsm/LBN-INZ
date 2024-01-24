'''
Plik zawiera funkcje do obliczania wyników strategii na podstawie predykcji modeli.
'''

import numpy as np
import pandas as pd

def get_close_pred(df_prediction, df, prefix, step):
    '''
    Function to transform prediction dataframe to one needed for strategy evaluation.
    
    Parameters:
    - df_prediction (pd.DataFrame): The input DataFrame with predictions.
    - df (pd.DataFrame): The input DataFrame with original data.
    - prefix (str): The prefix of the columns in df.
    - step (int): The step of prediction.
    
    Returns:
    - df_pred (pd.DataFrame): The output DataFrame with predictions.
    '''
    df_pred = df_prediction.copy()
    row_number_start = df.loc[df['Date'] == min(df_pred['Date'])].index[0]

    row_number_end = df.loc[df['Date'] == max(df_pred['Date'])].index[0]
    df_pred['Close_pred'] = np.exp(df_pred['pred']) *  df[prefix+'_Close'][row_number_start-step:row_number_end+1-step].values.flatten()
    df_pred['Close'] = df[prefix+'_Close'][row_number_start:row_number_end+1].values.flatten()
    df_pred['Open'] = df[prefix+'_Open'][(row_number_start):(row_number_end+1)].values.flatten()

    if step == 5:
        df_pred['Target'] = df['log_return_5'][row_number_start:row_number_end+1].values.flatten()
    elif step == 7:
        df_pred['Target'] = df['log_return_7'][row_number_start:row_number_end+1].values.flatten()

    df_pred['Price_buy'] = df[prefix+'_Open'][row_number_start-step+1:row_number_end-step+1+1].values.flatten()

    df_pred['Ratio_pred'] = ((df_pred['Close_pred'] - df_pred['Price_buy']) / df_pred['Price_buy'])*100

    return df_pred

def calculate_return(df, intrest, thr_p, thr_m):
    '''
    Function to calculate return of strategy.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with predictions.
    - intrest (float): The intrest rate.
    - thr_p (float): The threshold for positive prediction.
    - thr_m (float): The threshold for negative prediction.

    Returns:
    - return_mean (float): The mean return of strategy on given timeline, if there were any transactions otherwise -0.50.
    - return_list (list): The list of returns for each transaction made.
    '''

    return_list = []
    for _, row in df.iterrows():
            if row['Ratio_pred'] > thr_p:
                money_prc = (row['Close']*(1-intrest) - row['Price_buy']*(1+intrest))/row['Price_buy']
                return_list.append(money_prc)

            if row['Ratio_pred'] < thr_m:
                money_prc = (row['Price_buy']*(1-intrest) - row['Close']*(1+intrest))/row['Price_buy']
                return_list.append(money_prc)

    if return_list:
        return np.sum(return_list), return_list
    else:
        return -0.50, None