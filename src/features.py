import numpy as np

def add_time_features(df):
    x=df.copy(); x['hour']=x.index.hour; x['dayofweek']=x.index.dayofweek; x['is_weekend']=(x.dayofweek>=5).astype(int)
    x['hour_sin']=np.sin(2*np.pi*x.hour/24); x['hour_cos']=np.cos(2*np.pi*x.hour/24)
    x['dow_sin']=np.sin(2*np.pi*x.dayofweek/7); x['dow_cos']=np.cos(2*np.pi*x.dayofweek/7)
    return x

def make_supervised_table(df,target='Appliances'):
    x=add_time_features(df)
    for lag in [1,2,3,6,12,24,48,72,168]: x[f'lag_{lag}']=x[target].shift(lag)
    for w in [3,6,12,24,168]:
        s=x[target].shift(1); x[f'roll_mean_{w}']=s.rolling(w).mean(); x[f'roll_std_{w}']=s.rolling(w).std()
    return x.dropna()

def select_feature_columns(df,target='Appliances'): return [c for c in df.columns if c!=target]
