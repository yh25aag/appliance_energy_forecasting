import pandas as pd

def mean_forecast(y,h,index): return pd.Series(y.mean(),index=index,name='mean')
def naive_forecast(y,h,index): return pd.Series(y.iloc[-1],index=index,name='naive')
def seasonal_naive_forecast(y,h,index,seasonality):
    hist=list(y.values); vals=[]
    for _ in range(h): vals.append(hist[-seasonality]); hist.append(vals[-1])
    return pd.Series(vals,index=index)
def drift_forecast(y,h,index):
    slope=(y.iloc[-1]-y.iloc[0])/(len(y)-1)
    return pd.Series([y.iloc[-1]+slope*i for i in range(1,h+1)],index=index,name='drift')
