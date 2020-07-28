from trade import data
from datetime import datetime, timedelta
from datetime import date
import datetime
import pandas as pd 
import keras
from helper_functions import *
from model import Train_model



get = data('AAPL')
NAME = f"{get.identifier}_{date.today()}.h5"
print(NAME)
get.getdata()

x_train,x_test,y_train, y_test = get.get_clean_data()
print(x_train.shape)


# Train_model(x_train,y_train,x_test,y_test,NAME)

#new_model = keras.models.load_model("C:\\Users\\Asus\\Desktop\\stockpredict\\Saved_model\\{name}")

#print("-----------------------------------")
#print(new_model.predict(x_test))