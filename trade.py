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
from helper_functions import *




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

class data:
	def __init__(self,identifier):
		intrinio_sdk.ApiClient().configuration.api_key['api_key'] = 'OjYwYTkwN2VlMzc1YjM3Mjc4NWM0YmFjM2MxN2E4Mzgz'
		self.identifier = identifier
		self.start_date = days_ago
		self.end_date = today
		self.frequency = frequency
		self.page_size = page_size
		self.next_page = next_page

	
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
		finaldata['date'] = pd.to_datetime(finaldata['date'])
		self.testing_plot = finaldata
		finaldata.to_csv(f'D:\\PROJECTS\\python projects\\stockpredict\\Data\\{self.identifier}_daily_with_date.csv',index = False)
		finaldata.sort_values(by = 'date', axis = 0,ascending = True)
		finaldata.set_index('date',inplace =True)
		#print(finaldata.to_string())
		finaldata.to_csv(f'D:\\PROJECTS\\python projects\\stockpredict\\Data\\{self.identifier}_daily.csv',index = False)
		
		return finaldata

	def get_clean_data(self):
		data = pd.read_csv(f"D:\\PROJECTS\\python projects\\stockpredict\\Data\\{self.identifier}_daily.csv")
		data_normaliser = MinMaxScaler(feature_range  = (0,1))
		data_normalised = data_normaliser.fit_transform(data)
		x_normalised = np.array(data_normalised[:-1])
		x_normalised = np.delete(x_normalised, 3, axis=1)
		mask_x = np.median(x_normalised[x_normalised>0])
		x_normalised[x_normalised == 0] = mask_x
		y_normalised = np.array([item[3] for item in data_normalised])
		mask_y = np.median(y_normalised[y_normalised > 0])
		y_normalised[y_normalised == 0] = mask_y
		y_normalised = y_normalised[1::]
		X_train, X_test, Y_train, Y_test = train_test_split(x_normalised, y_normalised, test_size=0.2)
		X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
		X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
		print("X_train: "+ str(X_train.shape) +" Y_train: "+ str(Y_train.shape))
		print("X_test: "+ str(X_test.shape)+" Y_test: "+ str(Y_test.shape))
		length = len(x_normalised)
		print(f"length of dataset:{length}")
		return X_train,X_test,Y_train,Y_test	
		

	






	


		