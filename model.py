import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from PlotSplit import *
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def Train_model(x_train,y_train,x_test,y_test,name,Data):
	model = Sequential()
	model.add(LSTM(50,return_sequences = True,input_shape = (x_train.shape[1],1)))
	model.add(Dropout(0.2))

	model.add(LSTM(50,return_sequences = True))
	model.add(Dropout(0.2))

	model.add(LSTM(50,return_sequences = True))
	model.add(Dropout(0.2))

	model.add(LSTM(50,return_sequences = False))
	model.add(Dropout(0.2))

	model.add(Dense(1))


	model.compile(optimizer = 'adam',loss = 'mean_squared_error')
	model.fit(x_train,y_train, batch_size = 64, epochs = 100 ,validation_data = (x_test,y_test))
	model.summary()

	train_prediction = model.predict(x_train)
	test_prediction = model.predict(x_test)
	# print(train_prediction)
	# print(test_prediction)	
	scaler = MinMaxScaler(feature_range  = (0,1))
	external = scaler.fit(np.array(Data).reshape(-1,1))
	# external2 = scaler.fit(test_prediction)

	train_prediction = external.inverse_transform(train_prediction)
	test_prediction = external.inverse_transform(test_prediction)

	print("RMSE train_prediction: ",math.sqrt(mean_squared_error(y_train,train_prediction)))
	print("RMSE test_prediction: ",math.sqrt(mean_squared_error(y_test,test_prediction)))

	model.save(f'D:\\Github\\stockpredict\\Stock-forcast\\Saved_model\\{name}', save_format='h5')
	plot_data(Data,train_prediction,test_prediction)

	
	
	
	