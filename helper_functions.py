
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np 

	
def plot_raw_data(dataframe):
		plt.figure(figsize=(18,9))
		plt.plot('date','close' ,data = dataframe)
		plt.plot(dataframe['100ma'], color = 'red' )
		plt.xlabel('Date')
		plt.ylabel('Close Price')
		#plt.xticks(rotation=45)
		plt.show()


def plot_testing_data(dataframe):
	plt.figure(figsize = (18,9))
	dataframe = dataframe[801:-1]
	x = np.array(dataframe['date'])
	y = np.array(dataframe['close'])
	plt.plot(x,y)
	plt.xlabel('Date')
	plt.ylabel('Closeing Price')
	plt.xticks(rotation=45)
	plt.show()

def plot_validation(label,prediction):
	plt.figure(figsize = (18,9))
	label = np.array(label)
	prediction = np.array(prediction)
	plt.plot(label,prediction)
	plt.xlabel('Date')
	plt.ylabel('Closeing Price')
	plt.xticks(rotation=45)
	plt.show()
	







		
	

	






