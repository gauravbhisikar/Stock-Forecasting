import intrinio_sdk
from intrinio_sdk.rest import ApiException
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime, timedelta,date
import datetime 
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PlotSplit import *
import matplotlib.pyplot as plt



today = date.today()
days_ago = today - datetime.timedelta(days=1500)
frequency = 'daily'
page_size = 1001
next_page = ''
"""
identifier = 'AAPL' # str | A Security identifier (Ticker, FIGI, ISIN, CUSIP, Intrinio ID)
start_date = '2018-01-01' # date | Return prices on or after the date (optional)
end_date = '2019-01-01' # date | Return prices on or before the date (optional)
frequency = 'daily' # str | Return stock prices in the given frequency (optional) (default to daily)
page_size = 100 # int | The number of results to return (optional) (default to 100)
next_page = '' # str | Gets the next page of data from a previous API call (optional)

"""

class TrainingData:
	def __init__(self,identifier):
		intrinio_sdk.ApiClient().configuration.api_key['api_key'] = 'YOUR API KEY'
		self.identifier = identifier
		self.start_date = days_ago
		self.end_date = today
		self.frequency = frequency
		self.page_size = page_size
		self.next_page = next_page
		X = []
		self.original_data = X
	
	def getdata(self):
		security_api = intrinio_sdk.SecurityApi()
		try:
			api_response = security_api.get_security_stock_prices(self.identifier, start_date=self.start_date, end_date=self.end_date, frequency=self.frequency, page_size=self.page_size, next_page=self.next_page)
			data_frame = pd.DataFrame(api_response.stock_prices_dict)
			
		except ApiException as e:
			print("Exception when calling SecurityApi->get_security_stock_prices: %s\r\n" % e)
			print("wait for 1 min")
		
		
		finaldata = data_frame.filter(['date','open','high','low','close','volume','adj_open','adj_high','adj_low','adj_close','adj_volume'])
		finaldata['100ma']  = finaldata['close'].rolling(window = 100, min_periods=0).mean()
		self.date = finaldata['date']
		finaldata['date'] = pd.to_datetime(finaldata['date'])
		#plot_close(finaldata)

		finaldata.to_csv(f'D:\\PROJECTS\\python projects\\stockpredict\\Data\\{self.identifier}_daily_with_date.csv',index = False)
		
		# finaldata.sort_values(by = 'date', axis = 0,ascending = True)
		# finaldata.set_index('date',inplace =True)
		# #print(finaldata.to_string())

		# finaldata.to_csv(f'D:\\PROJECTS\\python projects\\stockpredict\\Data\\{self.identifier}_daily.csv',index = False)
		
		return finaldata

	def get_clean_data(self):
		data = pd.read_csv(f"D:\\PROJECTS\\python projects\\stockpredict\\Data\\{self.identifier}_daily_with_date.csv")
		print("Cleaning data  of ",self.identifier)
		X = data.filter(['close'])
		X = X.iloc[::-1]
		X = X.reset_index()['close']
		dataset = X.values
		# plt.plot(dataset)
		# plt.show()
		scaler = MinMaxScaler(feature_range  = (0,1))
		scaled_data = scaler.fit_transform(np.array(dataset).reshape(-1,1))
		train,test = split_dataset(scaled_data)
		X_train,Y_train = create_dataset(train,100)
		X_test,Y_test = create_dataset(test,100)
		# print(X_train.shape,Y_train.shape)
		# print(X_test.shape,Y_test.shape)
		

		X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
		X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
		# print("X_train: "+ str(X_train.shape) +" Y_train: "+ str(Y_train.shape))
		# print("X_test: "+ str(X_test.shape)+" Y_test: "+ str(Y_test.shape))
		return X_train,X_test,Y_train,Y_test,dataset

