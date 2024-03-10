import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from numpy import array

# creating a dataset matrix to train                      


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  # i=0, X=0,1,2,3-----99   Y=100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# LSTM models are sensitive to scale, so we scale the data using MinMaxScaler to make the values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
# file dir for Saving model
checkpoint_path = "300Epoch/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# Read CSV data
file_path = 'TSLA2.csv'
dataset = pd.read_csv(file_path)

# scaling the data
df1 = dataset.reset_index()['High']
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size

# splitting the dataset in to training and testing splits
train_data, test_data = df1[0:training_size,
                            :], df1[training_size:len(df1), :1]

time_step = 100
# reshaping input for the LSTM model
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Creating the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.load_weights(checkpoint_path).expect_partial()
# training the model
#model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=300,batch_size=64,verbose=1,callbacks=[cp_callback])
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
math.sqrt(mean_squared_error(y_train, train_predict))
math.sqrt(mean_squared_error(ytest, test_predict))
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2) +
                1:len(df1)-1, :] = test_predict
# getting the real prediction
x_input=test_data[341:].reshape(1,-1)
print(x_input.shape) # (1,305)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
# plot baseline and predictions
day_new=np.arange(1,101)
day_pred=np.arange(101,131)
scaler.inverse_transform(lst_output)
#plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
#plt.plot(day_pred,scaler.inverse_transform(lst_output))
#plt.plot(scaler.inverse_transform(df1))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
#plt.plot(scaler.inverse_transform(df1))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
plt.show()