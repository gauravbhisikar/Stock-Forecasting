import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from helper_functions import *

def Train_model(x_train,y_train,x_test,y_test,name):
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


	model.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics = ['accuracy'])
	model.fit(x_train,y_train, batch_size = 32, epochs = 10 ,validation_data = (x_test,y_test))
	model.summary()

	prediction = model.predict(x_test)
	print(prediction)
	
	scaler = StandardScaler()
	prediction = scaler.fit_transform(prediction)
	prediction = scaler.inverse_transform(prediction)
	print(prediction)
	x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
	plot_validation(x_test,prediction)

	model.save(f"C:\\Users\\Asus\\Desktop\\stockpredict\\Saved_model\\{name}")
	