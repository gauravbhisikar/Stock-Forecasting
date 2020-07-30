from CleanData import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from PlotSplit import *
from CleanData import TrainingData
from datetime import date
from model import Train_model


Ticker = 'MSFT'


Stock_Closing_Price = TrainingData(Ticker)

data = Stock_Closing_Price.getdata()


X = data.filter(['close'])
X = X.iloc[::-1]
X = X.reset_index()['close']
dataset = X.values

scaler = MinMaxScaler(feature_range  = (0,1))
external = scaler.fit(np.array(dataset).reshape(-1,1))

try:
	model = tf.keras.models.load_model(f'D:\\Github\\stockpredict\\Stock-forcast\\Saved_model\\{Ticker}.h5')
except:
	Stock_Closing_Price = TrainingData(Ticker)
	NAME = f"{Stock_Closing_Price.identifier}.h5"
	Stock_Closing_Price.getdata()
	x_train,x_test,y_train, y_test,original_data = Stock_Closing_Price.get_clean_data()
	Train_model(x_train,y_train,x_test,y_test,NAME,original_data)
	model = tf.keras.models.load_model(f'D:\\Github\\stockpredict\\Stock-forcast\\Saved_model\\{Ticker}.h5')




scaled_data = scaler.fit_transform(np.array(dataset).reshape(-1,1))


# print(scaled_data)



scaled_data = split_model_test_dataset(scaled_data)

# print(scaled_data)



scaled_data = create_model_test_dataset(scaled_data,100)
# print(scaled_data)



scaled_data =  np.reshape(scaled_data,(scaled_data.shape[0],scaled_data.shape[1],1))

predict = model.predict(scaled_data)
predict = external.inverse_transform(predict)


plot_predict(dataset,predict)

