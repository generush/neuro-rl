
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path

def filter_by_column_keywords(df, include_keywords, exclude_keyword):

    # Creating a mask for columns to include: columns that contain include_keywords or don't contain exclude_keyword
    filter_mask = df.columns.str.contains('|'.join(include_keywords)) | ~df.columns.str.contains(exclude_keyword)

    # Selecting columns based on the mask to get a filtered DataFrame
    return df.loc[:, filter_mask]

def process_data(file: str):
    return dd.read_parquet(file)

def process_csv(file: str):
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