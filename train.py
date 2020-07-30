from CleanData import TrainingData
from datetime import date
from PlotSplit import *
from model import Train_model


Stock_Closing_Price = TrainingData('AAPL')


NAME = f"{Stock_Closing_Price.identifier}.h5"


Stock_Closing_Price.getdata()


x_train,x_test,y_train, y_test,original_data = Stock_Closing_Price.get_clean_data()



Train_model(x_train,y_train,x_test,y_test,NAME,original_data)
