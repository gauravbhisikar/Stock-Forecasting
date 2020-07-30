
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np 
import matplotlib.dates as mdates
	



def plot_close(dataframe):
		plt.figure(figsize=(16,8))
		plt.title('Closing prices')
		plt.plot(dataframe['close'])
		plt.xlabel('Date',fontsize=18)
		plt.ylabel('Closing prices(INR)',fontsize=18)
		plt.show()



   

				

def split_dataset(dataframe):
	training_size=int(len(dataframe)*0.80)
	test_size=len(dataframe)-training_size
	train_data,test_data=dataframe[0:training_size,:],dataframe[training_size:len(dataframe),:1]
	print("training_size: ",training_size)
	print("test_size: ",test_size)
	return train_data,test_data	




def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)	





def plot_data(Original_data,train_prediction,test_prediction):
	train_plot = np.empty_like(Original_data)
	print(type(train_prediction))
	train_plot[:] = np.nan
	train_prediction = train_prediction.tolist()
	for i in range(0,len(train_prediction)):
		train_plot[i] = train_prediction[i][0]

	test_plot = np.empty_like(Original_data)	
	test_plot[:] = np.nan
	test_prediction = test_prediction.tolist()

	print("len Data",len(Original_data))
	print("train_prediction",len(train_prediction))
	print("test_prediction",len(test_prediction))
	print("len test_plot",len(test_plot))

	temp = 0
	for i in range(len(train_prediction),len(train_prediction)+len(test_prediction)):
		test_plot[i] = test_prediction[temp][0]
		temp +=1	

	
	plt.figure(figsize=(16,8))
	plt.title("Performance of Model on Training Data")
	plt.xlabel('Date')
	plt.ylabel('Closeing Price',fontsize=18)
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
	plt.gca().xaxis.set_major_locator(mdates.DayLocator())
	plt.plot(Original_data)
	plt.plot(train_plot)
	plt.plot(test_plot)
	plt.legend(['Original Data','Training Data prediction','Prediction'],loc='lower right')
	plt.show()




	







		
	

	






