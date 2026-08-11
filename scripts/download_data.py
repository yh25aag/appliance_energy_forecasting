from pathlib import Path; import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_download import download_raw_data,validate_raw_data
p=download_raw_data(); print(validate_raw_data(p).shape)
