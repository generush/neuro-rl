import pandas as pd
from constants.normalization_types import NormalizationType
from typing import List
from analysis.analyze_pca import compute_pca
from utils.io import import_pk, export_pk

def append_pc(df: pd.DataFrame, data_names: List[str], max_dims: int, norm_type: NormalizationType, column_label: str, export_path: str, tf_path: str = None) -> pd.DataFrame:

    pca_dict = {}
    df.reset_index(drop=True, inplace=True)  # Reset the index of the original DataFrame


    for data_type in data_names:
        
        # Modified line to exclude TIME_RAW field
        filt_data = df.loc[:, df.columns.str.contains(data_type + '_RAW') & ~df.columns.str.contains('TIME_RAW')].values

        if tf_path:
            scl = import_pk(tf_path + f'{data_type}_SCL.pkl')
            pca = import_pk(tf_path + f'{data_type}_PCA.pkl')
            n_dims = pca.n_components_
            data_normalized = scl.transform(filt_data)
            data_pc = pca.transform(data_normalized)
        else:
            n_dims = min(filt_data.shape[0], filt_data.shape[1], max_dims)
            scl, pca = compute_pca(n_dims, norm_type)
            data_normalized = scl.fit_transform(filt_data)
            data_pc = pca.fit_transform(data_normalized)


        if not tf_path:
            export_pk(scl, export_path + f'{data_type}_SCL.pkl')
            export_pk(pca, export_path + f'{data_type}_PCA.pkl')

            # Export cumulative variance
            pd.DataFrame(pca.explained_variance_ratio_.cumsum(), columns=['Cumulative Variance']).to_csv(export_path + 'info_' + data_type + '_cumvar.csv')

            # Export PCA components
            pd.DataFrame(pca.components_, columns=df.columns[df.columns.str.contains(data_type + '_RAW') & ~df.columns.str.contains('TIME_RAW')]).to_csv(export_path + 'PCA_components_' + data_type + '.csv')


        # create new df with column names
        pc_columns = [f'{data_type}_{column_label}_{i:03d}' for i in range(n_dims)]
        pc_df = pd.DataFrame(data_pc, columns=pc_columns).reset_index(drop=True)

        pca_dict[data_type] = {'scaler': scl, 'pca': pca}
        df = pd.concat([df, pc_df], axis=1)  # Append principal components to the original DataFrame

    return df, pca_dict