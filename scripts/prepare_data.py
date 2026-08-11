from pathlib import Path; import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_download import download_raw_data
from src.preprocessing import run_preprocessing
run_preprocessing(download_raw_data())
