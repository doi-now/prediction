#import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

#import data from yfinance
data = yf.download('DOGE-USD', start='2015-01-01', end='2023-01-01')

#create a dataframe and select only the 'Close' column
df = pd.DataFrame(data)
df = df[['Close']]

#convert dataframe to numpy array
dataset = df.values

#normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#split data into training and testing sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#reshape data into X_train and y_train datasets
X_train = []
y_train = []
for i in range(60, len(train)):
    X_train.append(train[i-60:i, 0])
    y_train.append(train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshape data into X_test and y_test datasets
X_test = []
y_test = []
for i in range(60, len(test)):
    X_test.append(test[i-60:i, 0])
    y_test.append(test[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

#reshape data into 3D for input into LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#fit the data to the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

#predicting the next 3 years of data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

#create a new dataframe to hold the predictions
pred_df = pd.DataFrame(predictions, columns=['Predicted Close'])

#create a datetime column for the predictions
pred_df['Date'] = data.index[-len(pred_df):]

#merge the predictions with the original dataframe
result_df = pd.concat([df, pred_df], axis=1)

#plot the results
plt.figure(figsize=(15,7))
plt.plot(df['Close'], label='Actual Price')
# plt.plot(result_df['Date'], result_df['Close'], label='Actual Price')
plt.plot(result_df['Date'], result_df['Predicted Close'], label='Predicted Price')
plt.title('BTC-USD Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()