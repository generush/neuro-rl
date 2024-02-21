# https://plotly.com/python/3d-scatter-plots/
import numpy as np
import pandas as pd

from utils.io import import_pk

def create_pc_data(FOLDER_PATH, FILE_NAME, DATASETS):

    # load DataFrame
    df = pd.read_parquet(FOLDER_PATH + FILE_NAME + '.parquet')

    new_dfs = []

    filt_pc_df = pd.DataFrame()  # List to store new DataFrames

    meta_df = df.loc[:, ~df.columns.str.contains('_RAW')]

    for idx, data_type in enumerate(DATASETS):

        # select data for PCA analysis (only raw data)
        filt_df = df.loc[:, df.columns.str.contains(data_type + '_RAW')]

        pca = import_pk(FOLDER_PATH + data_type + '_PCA' + '.pkl')
        scl = import_pk(FOLDER_PATH + data_type + '_SCL' + '.pkl')

        N_DIMENSIONS = pca.n_components

        if N_DIMENSIONS > 0:

            # create column name
            COLUMNS = np.char.mod(data_type + '_PC_%03d', np.arange(N_DIMENSIONS))

            # transform to PCA space
            filt_pc_data = pca.transform(scl.transform(filt_df.values))
            filt_pc_df = pd.DataFrame(data=filt_pc_data, columns=COLUMNS)

            new_dfs.append(filt_pc_df)

    new_df = pd.concat([meta_df] + new_dfs, axis=1)

    # Save the new DataFrame to a new parquet file
    new_df.to_parquet(FOLDER_PATH + 'PC_DATA_AVG' + '.parquet')
    new_df.to_csv(FOLDER_PATH + 'PC_DATA_AVG' + '.csv')

    return new_df
