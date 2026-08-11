from pathlib import Path
import pandas as pd
import requests

UCI_URL='https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv'
RAW_DIR=Path(__file__).resolve().parents[1]/'data/raw'
RAW_PATH=RAW_DIR/'energydata_complete.csv'

def download_raw_data(url=UCI_URL, output_path=RAW_PATH, force=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force: return output_path
    r=requests.get(url, timeout=60); r.raise_for_status(); output_path.write_bytes(r.content)
    return output_path

def validate_raw_data(path):
    df=pd.read_csv(path)
    required={'date','Appliances'}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {missing}')
    return df

if __name__=='__main__':
    p=download_raw_data(); df=validate_raw_data(p)
    print(df.shape); print(df.head())
