import pandas as pd
from scripts.train_utilities import window_dataset
from keras.models import load_model
import sys

def main(input_path, output_path, prefix):
    
    data = pd.read_csv(input_path)

    if prefix == "BA":
        
        window_size = 10

        target_cols = ['target_5']

        cols_used = [
                'norm_rsi_14', 'norm_slowk_14', 'minmax_daily_variation', 'minmax_BA_Volume',
                'mean_influential', 'mean_trustworthy', 'finbert_Score', 'bart_Score'
                ]
        
        model = load_model('models/'+prefix+'lstm_full_model.h5')

        step = 5

    elif prefix == "TSLA":
        
        window_size = 10

        target_cols = ['target_5']

        cols_used = [
                    'minmax_low_norm', 'minmax_high_norm', 'norm_rsi_gspc_14', 'norm_slowk_14' ,
                    'vader_Score', 'bart_Score', 'mean_influential', 'finbert_Score', 'mean_trustworthy'
                    ]
        
        model = load_model('models/'+prefix+'lstm_full_model.h5')

        step = 5

    elif prefix == "NFLX":

        window_size = 10

        target_cols = ['target_5']

        cols_used = [
                    'norm_rsi_gspc_14', 'norm_rsi_14',
                    'norm_slowk_14', 'minmax_high_norm', 'log_return_1'
                    ]
        
        model = load_model('models/'+prefix+'lstm_fin_model.h5')

        step = 5

    elif prefix == "BTC-USD":

        window_size = 14

        target_cols = ['target_7']

        cols_used = [
                    'minmax_BTC-USD_Volume', 'norm_rsi_14', 'norm_slowk_14', 'norm_rsi_gspc_14', 'minmax_daily_variation',
                    'finbert_Score', 'vader_Score', 'mean_influential'
                    ]
        
        model = load_model('models/'+prefix+'lstm_full_model.h5')

        step = 7

    else:
        print("Prefix not found")
        exit()

    X, _ = window_dataset(data[list(cols_used)+target_cols], target_cols[0], window_size)

    y_pred = model.predict(X)

    dates = data['Date']

    if step == 5:

        next_dates = pd.bdate_range(start=dates[-1], periods=step+1)[1:]
        dates_extended = dates.append(next_dates)
    
    else:

        next_dates = pd.date_range(start=dates[-1], periods=step+1)[1:]
        dates_extended = dates.append(next_dates)        

    df_pred = pd.DataFrame([y_pred.flatten(),dates_extended], columns=['pred','Date'])

    df_pred.to_csv(output_path+'/df_pred_'+prefix+'.csv', index=False)
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_path> <output_path> <prefix>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    prefix = sys.argv[3]

    main(input_path, output_path, prefix)

    #python model_pred.py /path/to/input.csv output_folder

