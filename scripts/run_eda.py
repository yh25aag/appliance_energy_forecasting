from pathlib import Path; import sys,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.eda import adf_test,save_eda
p=ROOT/'data/processed/appliance_hourly.csv'; df=pd.read_csv(p,parse_dates=['date'],index_col='date'); out=ROOT/'outputs/figures'; diff=save_eda(df,'Appliances',out); print('ADF original:',adf_test(df.Appliances)); print('ADF difference:',adf_test(diff))
