# https://plotly.com/python/3d-scatter-plots/
import numpy as np
import pandas as pd
from typing import List
from utils.metrics import tangling


# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

def append_tangling(df: pd.DataFrame, data_names: List[str], input_data_keyword: str) -> np.array:

    for data_type in data_names:

        # Select data for tangling analysis
        filt_data = df.loc[:,df.columns.str.contains(data_type + '_' + input_data_keyword)].values
        time = df['TIME'].values

        # Compute tangling
        np_tangling = tangling(filt_data, time)

        # Append tangling as a new column in the data DataFrame
        df_tangling = pd.DataFrame(np_tangling, columns = [data_type + '_TANGLING'])
        df = pd.concat([df, df_tangling], axis=1)

    # export DataFrame
    # data_w_tangling.to_parquet(path + 'RAW_DATA' + file_suffix + '_WITH_TANGLING' + '.parquet')
    # data_w_tangling.to_csv(path + 'RAW_DATA' + file_suffix + '_WITH_TANGLING' + '.csv')

    return df

