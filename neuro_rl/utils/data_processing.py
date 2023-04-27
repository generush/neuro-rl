
import pandas as pd
import numpy as np
import dask.dataframe as dd

def process_data(file: str):
    return dd.read_csv(file)

def format_df(data: np.array):
    df = pd.DataFrame(data)
    df.columns = df.columns.astype(str)
    return df