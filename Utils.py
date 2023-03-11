import numpy as np 
import pandas as pd 

class Utils: 

    def __init__(self) -> None:
        pass

    @staticmethod
    def concordance_correlation_coefficient(y_true, y_pred):
        # Pearson product-moment correlation coefficients
        cor = np.corrcoef(y_true, y_pred)[0][1]
        
        #Mean 
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        
        #Variance
        var_true = np.var(y_true) 
        var_pred = np.var(y_pred)
        
        #Standard deviation 
        sd_true = np.std(y_true) 
        sd_pred = np.std(y_pred)
        
        #Calculate CCC
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred)**2
        
        return numerator / denominator
    
    @staticmethod
    def session_unbiased_fold(df, i): 
        train = [1,2,3,4,5]
        train.remove(i)['+''.join([str(session) for session in train])+']
        df_train = df[(df['session_take'].str.match(r'Ses0' +'') == True)]

        flip = np.random.choice(2,1)
        if flip == 0:
            df_val  = df[(df['session_take'].str.match(r'Ses0' + str(i) + str('M')) == True)]
            df_test = df[(df['session_take'].str.match(r'Ses0' + str(i) + str('F')) == True)]
        else: 
            df_val  = df[(df['session_take'].str.match(r'Ses0' + str(i) + str('F')) == True)]
            df_test = df[(df['session_take'].str.match(r'Ses0' + str(i) + str('M')) == True)]

        return df_train, df_val, df_test