import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import numpy as np
import quandl
from keras.models import load_model
import streamlit as st

st.title('Stock Price Prediction')

user_input= st.text_input('Enter Stock Ticker','AAPL')
input='WIKI/'+user_input
df = quandl.get(input)

#Describing data
st.subheader('Data from 1980 to 2018')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()


fig= plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
plt.plot(ma200)
st.pyplot(fig)

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)

model= load_model('mymodel')

past_100_days= data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data= scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted= regressor.predict(x_test)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


