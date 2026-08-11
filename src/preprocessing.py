from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/energydata_complete.csv'
PROCESSED=ROOT/'data/processed/appliance_hourly.csv'

def load_raw_data(path=RAW):
    df=pd.read_csv(path)
    df['date']=pd.to_datetime(df['date'], errors='coerce')
    df=df.dropna(subset=['date']).set_index('date').sort_index()
    for c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
    return df

def missing_value_report(df):
    return pd.DataFrame({'missing_count':df.isna().sum(),'missing_percent':df.isna().mean()*100}).sort_values('missing_count',ascending=False)

def prepare_hourly_data(raw_df, output_path=PROCESSED):
    hourly=raw_df.resample('1h').mean()
    before=int(hourly.isna().sum().sum())
    hourly=hourly.interpolate('time',limit_direction='both').dropna()
    hourly.to_csv(output_path)
    print(f'Hourly rows: {len(hourly):,}; missing cells before interpolation: {before:,}')
    return hourly

def run_preprocessing(raw_path=RAW):
    raw=load_raw_data(raw_path)
    print('Raw shape:',raw.shape)
    print(missing_value_report(raw).head(10))
    return prepare_hourly_data(raw)
