import pandas as pd
import numpy as np
import datetime as dt
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA, ARMA
from tqdm import tqdm
def AutoArima(x):
    # return auto_arima(x)
    return auto_arima(  x, d=None, test = 'adf', D=1, m=24,
                        seasonal = True, error_action = 'ignore')

def Forecast(ARIMA_model, lastIndex, periods=48):
    # Forecast
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    # index_of_fc = pd.date_range(dataFrame.index[-1] + pd.DateOffset(months=1), periods = n_periods, freq='H')
    index_of_fc = pd.date_range(lastIndex + pd.DateOffset(hours=1), periods = n_periods, freq="H")
    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    return [fitted_series, lower_series, upper_series]

def processSARIMA():
#Read Time Series
    dataFrame = pd.read_excel("./data/DataAirTrain.xlsx",index_col='time', sheet_name="DataAirTrain")
    dataFrame =     dataFrame[[ 'NO','NO2','NOx','PM-1','PM-2-5','PM-10','TSP','RH','Temp',
                                'Barometer','Radiation','WindDir','SO2','Compass','CO','O3','Wind Spd',
                                'Hướng gió','Nhiệt độ','Áp suất khí quyển','Wind Spd (sai)']].tail(1000)
    
    # Compass and Radition have specific data 
    dataFrame[['Compass']] = dataFrame[['Compass']].interpolate(method='pad')
    dataFrame[['Radiation']] = dataFrame[['Radiation']].fillna(0)

    # Only NO2 is missing data too much so fill linear for other column
    dataFrame[['NO','NOx','PM-1','PM-2-5','PM-10','TSP','RH','Temp']] = dataFrame[['NO','NOx','PM-1','PM-2-5','PM-10','TSP','RH','Temp']].interpolate(method='linear')
    dataFrame[['WindDir','SO2','CO','O3','Wind Spd']] = dataFrame[['WindDir','SO2','CO','O3','Wind Spd']].interpolate(method='linear')

    # Using linear regression to fill missing data in NO2 column
    linearModel = LinearRegression()
    dataFrame = dataFrame.copy()
    nanMask = dataFrame[['NO2']].isna()
    nanMask = nanMask['NO2']

    xTrain = dataFrame.loc[~nanMask,['NOx']]
    yTrain = dataFrame.loc[~nanMask,['NO2']]

    linearModel.fit(xTrain, yTrain)

    xTest = dataFrame.loc[nanMask, ['NOx']]
    yTest = linearModel.predict(xTest)
    dataFrame.loc[nanMask,['NO2']] = yTest

    #Fix gap in time index
    dataFrame = dataFrame.resample('H').interpolate(method='linear')

    #Create data frames then fit to SARIMA model 
    len = 120
    step = 48
    predict = list()
    dataFrame = dataFrame.tail(len)
    for col in tqdm(dataFrame.columns.values):
        res = Forecast(AutoArima(dataFrame[col]), dataFrame.index[-1],step)
        predict.append(res[0])

    dateTimeIndex = pd.date_range(dataFrame.index[-1] + pd.DateOffset(hours=1), periods=step, freq="H")
    idx = pd.Index(dateTimeIndex, name='time')
    dataFramePredicted = pd.DataFrame(predict)
    dataFramePredicted = dataFramePredicted.transpose()
    dataFramePredicted.index = idx
    dataFramePredicted.columns = dataFrame.columns.values
    return dataFramePredicted
