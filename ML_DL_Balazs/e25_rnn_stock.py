import numpy as np
import pandas as pd 
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys

#------------ FETCHING AND PREPROCESSING THE DATA ----------------

prices_dataset_train =  pd.read_csv('data/SP500_train.csv')     # 1258
prices_dataset_test =  pd.read_csv('data/SP500_test.csv')       #   20

trainingset = prices_dataset_train['adj_close'].values.reshape(-1,1)
testset = prices_dataset_test['adj_close'].values.reshape(-1,1)
print(trainingset.shape)

# min-max normalization
min_max_scaler = MinMaxScaler(feature_range=(0,1))
scaled_trainingset = min_max_scaler.fit_transform(trainingset)

# create the training dataset because the features are the previous values
# so we have n previous values: and we predict the next value in the time series
X_train = []
y_train = []

# X             y
# 0 - 39        40
# 1 - 40        41
interval = 40
for i in range(interval, trainingset.shape[0]):
    # use the previous 40 prices in order to forecast the next one
    X_train.append(scaled_trainingset[i-interval:i,0])
    # indexes start with 0 so this is the target (the price tomorrow)
    y_train.append(scaled_trainingset[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)

# input shape for LSTM architecture
# reshape the dataset (numOfSamples, numOfFeatures, 1)
# 1 : want to predict the price tomorrow (so 1 value)
# numOfFeatures: the past prices we use as features
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)
print('')


#------------ BUILDING THE LSTM MODEL ----------------

# return sequence true because we have another LSTM after this one
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# RMSProp is working fine with LSTM but so do ADAM optimizer
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=1000, batch_size=128, verbose=1)


#------------ TESTING THE ALGORITHM ----------------
# training set + test set = 1258 + 20 = 1278
dataset_total = pd.concat((prices_dataset_train['adj_close'], prices_dataset_test['adj_close']), axis=0) #vertical axis=0 horizontal axis=1

# all inputs for test set
inputs = dataset_total[len(dataset_total)-len(prices_dataset_test)-interval:].values
inputs = inputs.reshape(-1,1)
inputs = min_max_scaler.transform(inputs)

print(inputs.shape,'\n')


X_test = []

for i in range(interval, len(prices_dataset_test)+interval):
    X_test.append(inputs[i-interval:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
print(X_test.shape,'\n')

predictions = model.predict(X_test)
print(predictions.shape,'\n')

# inverse the predicitons because we applied normalization but we want to compare with the original prices
predictions = min_max_scaler.inverse_transform(predictions)

# print(testset)
# print(predictions)

# plotting the results
plt.plot(testset, color='blue', label='Actual S&P500 Prices')
plt.plot(predictions, color='green', label='LSTM Predictions')
plt.title('S&P500 Predictions with Reccurent Neural Network')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
