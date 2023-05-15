
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


def process_data_to_pd(folder_path: str, file_suffix: str):

    data_dir = Path(folder_path)
    full_df = pd.concat(
        (pd.read_csv(file, index_col=False) for file in data_dir.glob('*' + file_suffix + '.csv')), axis=1
    )

    return full_df