import subprocess
import sys
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib

class MLService:
    def __init__(self):
        pass
    
    def predict(self, data):
        regressor = joblib.load('random_forest_model.pkl')
        scaler = StandardScaler()
        
        df = pd.DataFrame(data)
        
        df['EloDifference'] = df['White Elo'] - df['Black Elo']
        df['EloRatio'] = df['White Elo'] / df['Black Elo']
        df['RDdifference'] = df['White RD'] - df['Black RD']

        df_scaled = scaler.fit_transform(df)
        
        result_list = regressor.predict(df_scaled)
        result_list = result_list.tolist()
        
        return result_list