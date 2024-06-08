import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
except ImportError:
    install('pandas')
    import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

scaler = StandardScaler()

df = pd.read_csv('./data/2016_CvH.csv') 
df = df.dropna()

# get only the first 5000 rows
df = df.head(5000)

df['EloDifference'] = df['White Elo'] - df['Black Elo']
df['EloRatio'] = df['White Elo'] / df['Black Elo']
df['RDdifference'] = df['White RD'] - df['Black RD']

# Change Result-Winner to categorical
df['Result-Winner'] = df['Result-Winner'].astype('category').cat.codes

y = df['Result-Winner'].values # Target variable
selected_features = ['White Elo', 'Black Elo', 'White RD', 'Black RD', 'EloDifference', 'EloRatio', 'RDdifference']
X = df[selected_features].values

X_scaled = scaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(random_state=0, oob_score=True, n_estimators=200, max_depth=16, min_samples_split=2, min_samples_leaf=1, max_features='sqrt')

# Fit the regressor with x and y data
regressor.fit(X_train, y_train)

# Access the OOB Score
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')
 
# Making predictions on the same data or new data
predictions = regressor.predict(X_test)
allPredictions = regressor.predict(X_scaled)
 
# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')
 
r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2}')

joblib.dump(regressor, 'random_forest_model.pkl')