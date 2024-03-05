import pandas as pd
from constants.normalization_types import NormalizationType
from typing import List
from analysis.analyze_pca import compute_pca
from utils.io import export_pk

def append_pc(df: pd.DataFrame, data_names: List[str], max_dims: int, norm_type: NormalizationType, export_path: str) -> pd.DataFrame:

    pca_dict = {}

    for data_type in data_names:

        filt_data = df.loc[:,df.columns.str.contains(data_type + '_RAW')].values
        n_dims = min(filt_data.shape[0], filt_data.shape[1], max_dims)
        pc_columns = [f'{data_type}_PC_{i+1:03d}' for i in range(n_dims)]

        scl, pca, data_pc = compute_pca(filt_data, n_dims, norm_type)

        pc_df = pd.DataFrame(data_pc, columns=pc_columns)

        # Optionally export PCA and scaler objects
        export_pk(pca, export_path + f'{data_type}_PCA.pkl')
        export_pk(scl, export_path + f'{data_type}_SCL.pkl')

        pca_dict[data_type] = {'scaler': scl, 'pca': pca}
        df = pd.concat([df, pc_df], axis=1)  # Append principal components to the original DataFrame

    return df, pca_dict
