import pandas as pd
import numpy as np
import sys
from sklearn import metrics, linear_model
from sklearn.neighbors import KNeighborsRegressor
import warnings

# ignoring all of the pandas 'caveats in the documentation' warnings
warnings.filterwarnings("ignore")

filename = 'hly-temp-normal.txt'

# accepting sys.argv's in station names
test_set = []
for ind in range(1, len(sys.argv)):
    test_set.append(sys.argv[ind])

data = pd.read_csv(filename, delim_whitespace = True, header = None, na_values = ['-9999'])
data = data.replace({'[^0-9]*$':''}, regex=True)

# Made NA values equal to 0 so reset_index wouldn't delete the rows with NA values
data.fillna(0, inplace = True)

data.columns = ['name', 'month', 'day', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

data = data.set_index(['name','month','day'])

# Converted the data into features
data = data.stack()
data = data.reset_index().rename(columns = {0:'temp','level_3':'hour'})

# Refilled the 0's with NA's to use fillna function
data['temp'] = np.where(data['temp'] == 0, np.NaN, data['temp'])

# For values that are missing hour 0, use previous day hour 23 to fill
data['temp'] = data.groupby(['name'])['temp'].fillna(method ='ffill')

# Of the remaining with NA's it is assumed there are no values prior. Use next valid input
data['temp'] = data.groupby(['name'])['temp'].fillna(method = 'bfill')

# Check if there are still missing data
# data['temp'].isnull().any()

train_stations = []
test_stations = ['USW00023234', 'USW00014918', 'USW00012919', 'USW00013743','USW00025309']

names = data.groupby(['name'])
for name in names.groups:
    if name not in test_stations:
        train_stations.append(name)

# Training stations are ALWAYS the same regardless of user input for the sake of fitting the same model
train_data = data[data['name'].isin(train_stations)]

# Making test set based on user input
test_data = data[data['name'].isin(test_set)]

# The previous hour's temperature reading from the same station, if avail
def prev_hour(data):
    data['prev_hour'] = data.groupby(['name'])['temp'].shift(1).fillna(value = 0)
    #24 * 365 = 8760. Finding the last hour of 12/31 to use as first hour of 01/01
    data['lastday_temp'] = data.groupby(['name'])['temp'].shift(-8759)
    data['prev_hour'] = np.where(data['prev_hour'] == 0, data['lastday_temp'], data['prev_hour'])
    data.drop('lastday_temp', 1, inplace = True)
    return data

# The previous day's temperature reading at the same hour from the same station, if avail
def prev_day(data):
    data['prev_day'] = data.groupby(['name'])['temp'].shift(24).fillna(value = 0)
    data['lastday_temp'] = data.groupby(['name'])['temp'].shift(-8736)
    data['prev_day'] = np.where(data['prev_day'] == 0, data['lastday_temp'], data['prev_day'])
    data.drop('lastday_temp', 1, inplace=True)
    return data

# The mean temperature of that hour's reading (across all stations) on that day
def mean_temp_same_hour(data):
    data['temp'] = data['temp'].astype(int)
    data['mean_hour'] = data.groupby(['month', 'day', 'hour'])['temp'].transform(np.mean)
    return data

# The mean temperature for that day up to, but not including, the hour in question
def mean_temp_prior_hours(data):
    data['temp'] = data['temp'].astype(int)
    data['hour'] = data['hour'].astype(int)
    data['cum_sum'] = data.groupby(['name','month', 'day'])['temp'].cumsum().shift(1)
    data['mean_temp_prior'] = np.where(data['hour'] < 1, data['temp'], data['cum_sum']/data['hour'])
    data.drop('cum_sum', 1, inplace = True)
    return data

def findKNN(train_x, train_y, test_x, k):
    train_x = train_x.fillna(value = 0)
    test_x = test_x.fillna(value=0)
    algo = KNeighborsRegressor(n_neighbors = k)
    algo.fit(train_x, train_y)
    hypothesis = algo.predict(test_x)
    return hypothesis

def makeLR(train_x, train_y, test_x):
    train_x = train_x.fillna(value = 0)
    test_x = test_x.fillna(value=0)
    algo = linear_model.LinearRegression()
    algo.fit(train_x, train_y)
    hypothesis = algo.predict(test_x)
    return hypothesis

def calc_mse(targets, estimated):
    return metrics.mean_squared_error(targets, estimated)

features = ['prev_hour','prev_day', 'mean_hour', 'mean_temp_prior']

train_data_features = mean_temp_prior_hours(mean_temp_same_hour(prev_day(prev_hour(train_data))))

train_x = train_data_features[features].astype(float)
train_y = train_data_features[['temp']].astype(float)

test_data_features = mean_temp_prior_hours(mean_temp_same_hour(prev_day(prev_hour(test_data))))
test_x = test_data_features[features].astype(float)
actual_y = test_data_features[['temp']].astype(float)

print "CLIMATE KNN MSE:"
print calc_mse(actual_y, findKNN(train_x, train_y, test_x, 5))

print "CLIMATE LR MSE:"
print calc_mse(actual_y, makeLR(train_x, train_y, test_x))