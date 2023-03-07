import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
def subProcess(subDataFrame, step=48):
    model = VAR(subDataFrame)
    sorted_order = model.select_order(50)
    lag = sorted_order.aic
    modelFitted = model.fit(lag)

    #Forcasting
    predictions = modelFitted.forecast(subDataFrame.values[-lag :], steps = step)

    #Parse result to data frame
    dateTimeIndex = pd.date_range(subDataFrame.index[-1] + pd.DateOffset(hours=1), periods=step, freq="H")
    idx = pd.Index(dateTimeIndex, name='time')
    dataFramePredicted = pd.DataFrame(predictions, index=idx, columns=subDataFrame.columns.values)
    return dataFramePredicted

def processVAR():
    #Read Time Series
    dataFrame = pd.read_excel("./data/DataAirTrain.xlsx",index_col='time', sheet_name="DataAirTrain")
    dataFrame = dataFrame[['NO','NO2','NOx','PM-1','PM-2-5','PM-10','TSP','RH','Temp']].tail(1000)

    #Only NO2 is missing data too much so fill linear for other column
    dataFrame[['NO','NOx','PM-1','PM-2-5','PM-10','TSP','RH','Temp']] = dataFrame[['NO','NOx','PM-1','PM-2-5','PM-10','TSP','RH','Temp']].interpolate(method='linear')

    #Using linear regression to fill missing data in NO2 column
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

    #Create 3 data frame then fit to 3 VAR model 
    step = 48
    listOfDataFrame = list()
    listOfDataFrame.append(subProcess(dataFrame[['NO','NO2','NOx']], step))
    listOfDataFrame.append(subProcess(dataFrame[['PM-1','PM-2-5','PM-10','TSP']], step))
    listOfDataFrame.append(subProcess(dataFrame[['RH','Temp']], step))

    resultPrediction = pd.concat(listOfDataFrame, axis=1, ignore_index=False)
    return resultPrediction
