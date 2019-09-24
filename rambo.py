import pandas as pd 
import datetime
from matplotlib import pyplot as plt
import fbprophet #forecasting time series data based on an additive model 

#To upload the file to the Google Colab 
from google.colab import files
uploaded = files.upload()

#Reads the csv file
data=pd.read_csv('Power-Networks-LCL.csv',parse_dates=["DateTime"])
data #displays the content of the file


#To plot the Dataset in weekly formate
week=data.resample('w').sum()
plt.figure(figsize=(8,5))
plt.plot(week)
plt.show()

# Find Top3 household that has more samples
top3= data.LCLid.value_counts().to_frame().head(3)
top3

for index, row in top3.iterrows():
  
    dph = data[data["LCLid"] == index]
    daily=dph.resample('D',on='DateTime').sum()
    data2=daily
	
	  #Before using Prophet, we rename the columns as "ds" and "y"
    data2.reset_index(inplace=True)
    data2=data2.rename(columns={'DateTime':'ds','KWh':'y'})
	
    #Make the prophet model and fit on the data
    data2_prophet=fbprophet.Prophet(changepoint_prior_scale=0.10)
    data2_prophet.fit(data2)
    
    print('Prediction for ',index)
    
    #4 month future dataframe
    data2_forecast=data2_prophet.make_future_dataframe(periods=30*4,freq="D")
    # make prediction
    data2_forecast=data2_prophet.predict(data2_forecast)
	
  	#plots the forecasted data
    data2_prophet.plot(data2_forecast,xlabel='Date',ylabel='KWh')
    plt.title('Forecasted value')
    data2_prophet.plot_components(data2_forecast)
