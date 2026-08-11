import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y,p): return np.sqrt(mean_squared_error(y,p))
def smape(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); d=np.abs(y)+np.abs(p)
    return 100*np.mean(np.where(d==0,0,2*np.abs(p-y)/d))
def mase(y,p,train,s=24):
    e=np.abs(np.asarray(train)[s:]-np.asarray(train)[:-s]).mean()
    return np.mean(np.abs(np.asarray(y)-np.asarray(p)))/e if e else np.nan
def evaluate(name,y,p,train): return {'model':name,'MAE':mean_absolute_error(y,p),'RMSE':rmse(y,p),'sMAPE':smape(y,p),'MASE':mase(y,p,train),'Bias':np.mean(np.asarray(p)-np.asarray(y))}
def compare(forecasts,test,train):
    import pandas as pd
    return pd.DataFrame([evaluate(n,test.loc[v:=pd.Series(p,index=test.index).notna()],pd.Series(p,index=test.index).loc[v],train) for n,p in forecasts.items()]).sort_values('RMSE').reset_index(drop=True)
