from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Union
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
import umap

from utils.data_processing import process_data, format_df

@dataclass
class Data:
    raw: pd.DataFrame
    # cos: pd.DataFrame = field(init=False)

    # def __post_init__(self):
        # self.cos = pd.DataFriame(sklearn.metrics.pairwise.cosine_distances(self.raw.compute()))

@dataclass
class MultiDimensionalScalingEmbedding:
    def fit_transform(self, x: np.array) -> np.array:
        # TODO: GENE
        return x

@dataclass
class Embedding(ABC):
    """Abstract base class for embedding"""

    @abstractmethod
    def fit_transform(self, x):
        """fit and transform data using embedding"""

@dataclass
class PCAEmbedding(Embedding):
    embedding: sklearn.decomposition.PCA
    x_embd: pd.DataFrame = field(init=False)

    def fit_transform(self, x: Data) -> np.array:
        self.x_embd = format_df(self.embedding.fit_transform(x.raw))

    @property
    def var(self) -> np.array:
        return self.embedding.explained_variance_ratio_

    @property
    def cumvar(self) -> np.array:
        return np.cumsum(self.embedding.explained_variance_ratio_)

@dataclass
class MDSEmbedding(Embedding):
    """Classical Multidimensional Scaling (MDS) class"""
    embedding: MultiDimensionalScalingEmbedding
    x_embd: pd.DataFrame = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.cos)
        self.x_embed = format_df(x1 / np.max(x1))

@dataclass
class ISOMAPEmbedding(Embedding):
    """Isomap (ISOMAP) Class"""
    embedding: sklearn.manifold.Isomap
    x_embd: pd.DataFrame = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.cos)
        self.x_embd = format_df(x1 / np.max(x1))

@dataclass
class LLEEmbedding(Embedding):
    """Locally linear embedding (LLE) class"""
    embedding: sklearn.manifold.LocallyLinearEmbedding
    x_embd: pd.DataFrame = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.cos)
        self.x_embd = format_df(x1 / np.max(x1))

@dataclass
class LEMEmbedding(Embedding):
    """Laplacian Eigenmaps (LEM) class"""
    embedding: sklearn.manifold.SpectralEmbedding
    x_embd: pd.DataFrame = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.cos)
        self.x_embd = format_df(x1 / np.max(x1))

@dataclass
class TSNEEmbedding(Embedding):
    """t-distributed Stocastic Neighbor Embedding (t-SNE) class"""
    embedding: sklearn.manifold.TSNE
    x_embd: pd.DataFrame = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.raw)
        self.x_embd = format_df(x1 / np.max(x1))

def center_scale(x: np.array) -> np.array:
    xc = x - np.mean(x)
    return xc/np.max(xc)

@dataclass
class UMAPEmbedding(Embedding):
    """Uniform Embedding Approximation and Projection (UMAP) class"""
    embedding: umap.UMAP
    x_embd: pd.DataFrame = field(init=False)

    def fit_transform(self, x: Data) -> None:
        self.x_embd = format_df(center_scale(self.embedding.fit_transform(x.raw)))

DIMS = None # 5

@dataclass
class Embeddings:
    data: Data
    embeddings: Dict[str, Union[PCAEmbedding, MDSEmbedding, ISOMAPEmbedding, LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding]]

    def __post_init__(self):
        for key in self.embeddings:
            self.embeddings[key].fit_transform(self.data)
