
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path

def process_data(file: str):
    return dd.read_csv(file)

def format_df(data: np.array):
    df = pd.DataFrame(data)
    df.columns = df.columns.astype(str)
    return df


def process_data_to_pd(file_path: str):

    data_dir = Path(file_path)
    full_df = pd.concat(
        (pd.read_csv(file, index_col=0) for file in data_dir.glob('*DATA.csv')), axis=1
    )
    # full_df.to_csv(file_path + 'csv_file.csv')

    return full_df