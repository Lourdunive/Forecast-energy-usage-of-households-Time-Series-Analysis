import pandas as pd 
import datetime
from matplotlib import pyplot as plt
import numpy as np
import fbprophet #forecasting time series data based on an additive model 

#To upload the file to the Google Colab 
from google.colab import files
uploaded = files.upload()

#Reads the csv file
data=pd.read_csv('Power-Networks-LCL.csv',parse_dates=["DateTime"],index_col="DateTime")
data #displays the content of the file

# Find Top3 household that has more samples
top3 = data.LCLid.value_counts().to_frame().head(3)
print(top3)

#To plot the Dataset in weekly formate
week=data.resample('w').sum()
plt.figure(figsize=(8,5))
plt.plot(week)
plt.show()

#plots groupby and mean function for hourly summary
hr_time=data.groupby(data.index.time).mean()
hr_ticks=4*60*60* np.arange(6)
hr_time.plot(xticks=hr_ticks,figsize=(6,5), linewidth=3)
plt.show()

#Before using Prophet, we rename the columns as "ds" and "y"
daily=data.resample('D').sum()
data2=daily
data2.reset_index(inplace=True)
data2=data2.rename(columns={'DateTime':'ds','KWh':'y'})
data2.head()

#Make the prophet model and fit on the data
data2_prophet=fbprophet.Prophet(changepoint_prior_scale=0.10)
data2_prophet.fit(data2)

# 4 month future dataframe
data2_forecast=data2_prophet.make_future_dataframe(periods=30*4,freq="D")
# makes prediction
data2_forecast=data2_prophet.predict(data2_forecast)

#plots the forcasted data
data2_prophet.plot(data2_forecast,xlabel='Date',ylabel='KWh')
plt.title('prediction for 4 months ')
plt.show()

#visualize the overall trend and the component patterns
data2_prophet.plot_components(data2_forecast)
plt.show()
