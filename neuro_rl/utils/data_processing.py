
import pandas as pd
import numpy as np

def process_data(file: str):
    # read csv file        
    raw_data = pd.read_csv(file)
    # remove extra rows of zeros
    clean_data = raw_data.loc[~(raw_data==0).all(axis=1)]
    clean_data.columns = clean_data.columns.astype(str)
    return clean_data

def format_df(data: np.array):
    df = pd.DataFrame(data)
    df.columns = df.columns.astype(str)
    return df