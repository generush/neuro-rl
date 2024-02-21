import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from constants.normalization_types import NormalizationType
from .scalers import RangeScaler, RangeSoftScaler

def normalize(data: np.array, norm_type: NormalizationType) -> pd.DataFrame:
    """
    Normalizes the DataFrame based on the specified normalization type.
    """
    if norm_type == NormalizationType.Z_SCORE.value:
        scaler = StandardScaler()
    elif norm_type == NormalizationType.RANGE.value:
        scaler = RangeScaler()
    elif norm_type == NormalizationType.RANGE_SOFT.value:
        scaler = RangeSoftScaler(softening_factor=5)  # Adjust the softening factor as needed
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    data_out = scaler.fit_transform(data)
    return data_out, scaler

def transform_pc(data: np.array, n_components: int) -> pd.DataFrame:
    """
    Performs PCA on the normalized DataFrame and returns the principal components.
    """
    pca = PCA(n_components=n_components)
    data_pc = pca.fit_transform(data)
    return data_pc, pca