# https://plotly.com/python/3d-scatter-plots/
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Union

import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
import umap
from dash import Dash, Input, Output, dcc, html

# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-02/20-36-21/arnn.csv' # arnn_pc
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-09/14-21-16/obs.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-09/14-21-16/obs_pc.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-09/14-21-16/act.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-09/14-21-16/act_pc.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-09/14-21-16/arnn.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-09/14-21-16/arnn_pc.csv'
# # DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-09/14-21-16/crnn.csv'
# # DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-09/14-21-16/crnn_pc.csv'

# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-15/19-07-42/obs.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-15/19-07-42/obs_pc.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-15/19-07-42/act.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-15/19-07-42/act_pc.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-15/19-07-42/arnn.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-15/19-07-42/arnn_pc.csv'
# # DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-15/19-07-42/crnn.csv'
# DATA_PATH = '/home/gene/Downloads/crl/outputs/2023-02-15/19-07-42/crnn_pc.csv'

# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data/hxs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data/obs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data/act.csv'

# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2023_03_09/obs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2023_03_09/act.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2023_03_09/cxs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2023_03_09/hxs.csv'

# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2023_03_09/obs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2023_03_09/act.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2023_03_09/cxs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2023_03_09/hxs.csv'

# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/anymalterrain_2lstm_data_2023_03_09/actor_rnn/obs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/anymalterrain_2lstm_data_2023_03_09/actor_rnn/act.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/anymalterrain_2lstm_data_2023_03_09/actor_rnn/cxs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/anymalterrain_2lstm_data_2023_03_09/actor_rnn/hxs.csv' # WHOA MASSIVE AND UNIQUE TRAJECTORIES! 80% in PC1

# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/anymalterrain_2lstm_data_2023_03_09/critic_rnn/obs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/anymalterrain_2lstm_data_2023_03_09/critic_rnn/act.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/anymalterrain_2lstm_data_2023_03_09/critic_rnn/cxs.csv' # YESSSSSS Unique path trajectoty!
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/anymalterrain_2lstm_data_2023_03_09/critic_rnn/hxs.csv' # CAN'T SEE IN PLOT, 97% in PC1


# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2lstm_2023_03_10/actor_rnn/obs.csv' # looks similar to actor cxs, 35% explained by PC1/2, 97% by PC20
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2lstm_2023_03_10/actor_rnn/act.csv' # looks similar to actor cxs, 35% explained by PC1/2, 100% by PC20
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2lstm_2023_03_10/actor_rnn/cxs.csv' # orbits with high variability (moreso than anymal makes sense). 31% explained by PC1/2, 75% from 20PC
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2lstm_2023_03_10/actor_rnn/hxs.csv' # CAN'T SEE

# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2lstm_2023_03_10/critic_rnn/obs.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2lstm_2023_03_10/critic_rnn/act.csv'
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2lstm_2023_03_10/critic_rnn/cxs.csv' # Cluster but seemingly unique path? 45% explained by PC1/2, 80% from 20PC
# DATA_PATH = '/home/gene/code/IsaacGymEnvs/isaacgymenvs/shadowhand_data_2lstm_2023_03_10/critic_rnn/hxs.csv' # CAN'T See

########################################################### Created Submodules ###################################################

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_2games/obs.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_2games/act.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_2games/acx.csv' # ORBIT
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_2games/ahx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_2games/ccx.csv' # DISTINCT CLUSTER/PATH
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_2games/chx.csv'

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_100games/obs.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_100games/act.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_100games/acx.csv' # BANANA
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_100games/ahx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_100games/ccx.csv' # TIGHT CLUSTER/PATH
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/anymalterrain_100games/chx.csv'

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/shadowhand/obs.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/shadowhand/act.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/shadowhand/acx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/shadowhand/ahx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/shadowhand/ccx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/shadowhand/chx.csv'

############################## Made 53 second video (30FPS = 1599 frames) #############################

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-08-22/obs.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-08-22/act.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-08-22/acx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-08-22/ahx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-08-22/ccx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-08-22/chx.csv'

######################## With rewards (signifying reached the block goal pose) ##########################

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-54-15/obs.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-54-15/act.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-54-15/rew.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-54-15/dne.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-54-15/acx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-54-15/ahx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-54-15/ccx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-11_15-54-15/chx.csv'

######################## Separated actions from obs vector, corrected hn and cn (were swapped before) ##########################

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/obs.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/act.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/rew.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/dne.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/acx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/ahx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/ccx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/chx.csv'

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/obs.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/act.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/rew.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/dne.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/acx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/ahx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/ccx.csv'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/chx.csv'

######################## Loading all data files ##########################

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/'

DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/'

def process_data(dic: str):
    # read csv file        
    raw_data = pd.read_csv(DATA_PATH + dic + '.csv')
    # remove extra rows of zeros
    clean_data = raw_data.loc[~(raw_data==0).all(axis=1)]
    return clean_data

@dataclass
class Data:
    raw: np.array
    cos: np.array = field(init=False)

    def __post_init__(self):
        self.cos = sklearn.metrics.pairwise.cosine_distances(self.raw)

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
    x_embd: np.array = field(init=False)

    def fit_transform(self, x: Data) -> np.array:
        self.x_embd = self.embedding.fit_transform(x.raw)

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
    x_embd: np.array = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.cos)
        self.x_embed = x1 / np.max(x1)

@dataclass
class ISOMAPEmbedding(Embedding):
    """Isomap (ISOMAP) Class"""
    embedding: sklearn.manifold.Isomap
    x_embd: np.array = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.cos)
        self.x_embd = x1 / np.max(x1)

@dataclass
class LLEEmbedding(Embedding):
    """Locally linear embedding (LLE) class"""
    embedding: sklearn.manifold.LocallyLinearEmbedding
    x_embd: np.array = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.cos)
        self.x_embd = x1 / np.max(x1)

@dataclass
class LEMEmbedding(Embedding):
    """Laplacian Eigenmaps (LEM) class"""
    embedding: sklearn.manifold.SpectralEmbedding
    x_embd: np.array = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.cos)
        self.x_embd = x1 / np.max(x1)

@dataclass
class TSNEEmbedding(Embedding):
    """t-distributed Stocastic Neighbor Embedding (t-SNE) class"""
    embedding: sklearn.manifold.TSNE
    x_embd: np.array = field(init=False)

    def fit_transform(self, x: Data) -> None:
        x1 = self.embedding.fit_transform(x.raw)
        self.x_embd = x1 / np.max(x1)

def center_scale(x: np.array) -> np.array:
    xc = x - np.mean(x)
    return xc/np.max(xc)

@dataclass
class UMAPEmbedding(Embedding):
    """Uniform Embedding Approximation and Projection (UMAP) class"""
    embedding: umap.UMAP
    x_embd: np.array = field(init=False)

    def fit_transform(self, x: Data) -> None:
        self.x_embd = center_scale(self.embedding.fit_transform(x.raw))

DIMS = 5

@dataclass
class Embeddings:
    data: Data
    embeddings: Dict[str, Union[PCAEmbedding, MDSEmbedding, ISOMAPEmbedding, LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding]]

    def __post_init__(self):
        for key in self.embeddings:
            self.embeddings[key].fit_transform(self.data)

# @dataclass
# class Embeddings:
#     data: np.array
#     pca: Embedding
#     mds: Embedding = None
#     iso: Embedding = None
#     lle: Embedding = None
#     lem: Embedding = None
#     tsne: Embedding = None
#     umap: Embedding = None

obs = Embeddings(
    data=Data(process_data('obs')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
        'mds': MDSEmbedding(MultiDimensionalScalingEmbedding()),
        'iso': ISOMAPEmbedding(sklearn.manifold.Isomap(n_components=DIMS, n_neighbors=40)),
        'lle': LLEEmbedding(sklearn.manifold.LocallyLinearEmbedding(n_components=DIMS, n_neighbors=60, method='modified')),
        'lem': LEMEmbedding(sklearn.manifold.SpectralEmbedding(n_components=DIMS, affinity='nearest_neighbors', n_neighbors=60)),
        'tsne': TSNEEmbedding(sklearn.manifold.TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42)),
        'umap': UMAPEmbedding(umap.UMAP(n_components=DIMS, metric='cosine', n_neighbors=70, random_state=42))
    }
)
logging.info('Finished computing obs embedding')

act = Embeddings(
    data=Data(process_data('act')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
        'mds': MDSEmbedding(MultiDimensionalScalingEmbedding()),
        'iso': ISOMAPEmbedding(sklearn.manifold.Isomap(n_components=DIMS, n_neighbors=40)),
        'lle': LLEEmbedding(sklearn.manifold.LocallyLinearEmbedding(n_components=DIMS, n_neighbors=60, method='modified')),
        'lem': LEMEmbedding(sklearn.manifold.SpectralEmbedding(n_components=DIMS, affinity='nearest_neighbors', n_neighbors=60)),
        'tsne': TSNEEmbedding(sklearn.manifold.TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42)),
        'umap': UMAPEmbedding(umap.UMAP(n_components=DIMS, metric='cosine', n_neighbors=70, random_state=42))
    }
)
logging.info('Finished computing act embedding')

acx = Embeddings(
    data=Data(process_data('acx')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
        'mds': MDSEmbedding(MultiDimensionalScalingEmbedding()),
        'iso': ISOMAPEmbedding(sklearn.manifold.Isomap(n_components=DIMS, n_neighbors=40)),
        'lle': LLEEmbedding(sklearn.manifold.LocallyLinearEmbedding(n_components=DIMS, n_neighbors=60, method='modified')),
        'lem': LEMEmbedding(sklearn.manifold.SpectralEmbedding(n_components=DIMS, affinity='nearest_neighbors', n_neighbors=60)),
        'tsne': TSNEEmbedding(sklearn.manifold.TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42)),
        'umap': UMAPEmbedding(umap.UMAP(n_components=DIMS, metric='cosine', n_neighbors=70, random_state=42))
    }
)
logging.info('Finished computing acx embedding')

ahx = Embeddings(
    data=Data(process_data('ahx')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
        'mds': MDSEmbedding(MultiDimensionalScalingEmbedding()),
        'iso': ISOMAPEmbedding(sklearn.manifold.Isomap(n_components=DIMS, n_neighbors=40)),
        'lle': LLEEmbedding(sklearn.manifold.LocallyLinearEmbedding(n_components=DIMS, n_neighbors=60, method='modified')),
        'lem': LEMEmbedding(sklearn.manifold.SpectralEmbedding(n_components=DIMS, affinity='nearest_neighbors', n_neighbors=60)),
        'tsne': TSNEEmbedding(sklearn.manifold.TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42)),
        'umap': UMAPEmbedding(umap.UMAP(n_components=DIMS, metric='cosine', n_neighbors=70, random_state=42))
    }
)
logging.info('Finished computing ahx embedding')

ccx = Embeddings(
    data=Data(process_data('ccx')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
        'mds': MDSEmbedding(MultiDimensionalScalingEmbedding()),
        'iso': ISOMAPEmbedding(sklearn.manifold.Isomap(n_components=DIMS, n_neighbors=40)),
        'lle': LLEEmbedding(sklearn.manifold.LocallyLinearEmbedding(n_components=DIMS, n_neighbors=60, method='modified')),
        'lem': LEMEmbedding(sklearn.manifold.SpectralEmbedding(n_components=DIMS, affinity='nearest_neighbors', n_neighbors=60)),
        'tsne': TSNEEmbedding(sklearn.manifold.TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42)),
        'umap': UMAPEmbedding(umap.UMAP(n_components=DIMS, metric='cosine', n_neighbors=70, random_state=42))
    }
)
logging.info('Finished computing ccx embedding')

chx = Embeddings(
    data=Data(process_data('chx')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
        'mds': MDSEmbedding(MultiDimensionalScalingEmbedding()),
        'iso': ISOMAPEmbedding(sklearn.manifold.Isomap(n_components=DIMS, n_neighbors=40)),
        'lle': LLEEmbedding(sklearn.manifold.LocallyLinearEmbedding(n_components=DIMS, n_neighbors=60, method='modified')),
        'lem': LEMEmbedding(sklearn.manifold.SpectralEmbedding(n_components=DIMS, affinity='nearest_neighbors', n_neighbors=60)),
        'tsne': TSNEEmbedding(sklearn.manifold.TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42)),
        'umap': UMAPEmbedding(umap.UMAP(n_components=DIMS, metric='cosine', n_neighbors=70, random_state=42))
    }
)
logging.info('Finished computing chx embedding')


# Generate random data for x, y, and z coordinates
n_points = 100
x = obs.embeddings['pca'].x_embd[:,0]
y = obs.embeddings['pca'].x_embd[:,1]
z = obs.embeddings['pca'].x_embd[:,2]

# Create a 3D figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(x, y, z, c=z, marker='o')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Add lines connecting the points
for i in range(n_points-1):
    ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], c='gray')

# Add a color bar to the plot
cbar = fig.colorbar(scatter)
cbar.set_label('Z Value')

# Show the plot
plt.show()





import matplotlib.pyplot as plt

# Generate random data for x, y, and z coordinates
n_points = 100
x = act.embeddings['pca'].x_embd[:,0]
y = act.embeddings['pca'].x_embd[:,1]
z = act.embeddings['pca'].x_embd[:,2]

# Create a 3D figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(x, y, z, c=z, marker='o')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()





import matplotlib.pyplot as plt

# Generate random data for x, y, and z coordinates
n_points = 100
x = acx.embeddings['pca'].x_embd[:,0]
y = acx.embeddings['pca'].x_embd[:,1]
z = acx.embeddings['pca'].x_embd[:,2]

# Create a 3D figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(x, y, z, c=z, marker='o')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()



import matplotlib.pyplot as plt

# Generate random data for x, y, and z coordinates
n_points = 100
x = obs.embeddings['pca'].x_embd[:,0]
y = obs.embeddings['pca'].x_embd[:,1]
z = obs.embeddings['pca'].x_embd[:,2]

# Create a 3D figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(x, y, z, c=z, marker='o')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()



import matplotlib.pyplot as plt

# Generate random data for x, y, and z coordinates
n_points = 100
x = ccx.embeddings['pca'].x_embd[:,0]
y = ccx.embeddings['pca'].x_embd[:,1]
z = ccx.embeddings['pca'].x_embd[:,2]

# Create a 3D figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(x, y, z, c=z, marker='o')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()



import matplotlib.pyplot as plt

# Generate random data for x, y, and z coordinates
n_points = 100
x = chx.embeddings['pca'].x_embd[:,0]
y = chx.embeddings['pca'].x_embd[:,1]
z = chx.embeddings['pca'].x_embd[:,2]

# Create a 3D figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(x, y, z, c=z, marker='o')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()









n_steps = obs.data.raw.shape[0]

# This was helpful
# https://stackoverflow.com/questions/69647738/dash-output-multiple-graph-based-on-users-graph-choice

# https://community.plotly.com/t/dash-bootstrap-components-grid-system-not-working/30957
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.P("Time window width:"),
                            dcc.Slider(
                                id='time-window-slider',
                                min=0,
                                max=n_steps,
                                step=1,
                                value=n_steps,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.P("Time start:"),
                            dcc.Slider(
                                id='time-start-slider',
                                min=0,
                                max=n_steps,
                                step=1,
                                value=0,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                    ),
                    align="center",
                ),
            ],
            # Freeze top row with the time range sliders
            # https://community.plotly.com/t/freeze-one-row-of-dash-using-dashbootstrap/71375/9
            className='sticky-top', style={'position': 'sticky-top'},
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-obs-pc', options=pd.DataFrame(obs.embeddings[0].x_embd).columns.values, value=pd.DataFrame(obs.embeddings[0].x_embd).columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-obs-pc', options=pd.DataFrame(obs.embeddings[0].x_embd).columns.values, value=pd.DataFrame(obs.embeddings[0].x_embd).columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-obs-pc', options=pd.DataFrame(obs.embeddings[0].x_embd).columns.values, value=pd.DataFrame(obs.embeddings[0].x_embd).columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-obs-pc"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-obs-cln', options=data['obs']['cln'].columns.values, value=data['obs']['cln'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-obs-cln', options=data['obs']['cln'].columns.values, value=data['obs']['cln'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-obs-cln', options=data['obs']['cln'].columns.values, value=data['obs']['cln'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-obs-cln"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-act-pc', options=data['act']['pc'].columns.values, value=data['act']['pc'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-act-pc', options=data['act']['pc'].columns.values, value=data['act']['pc'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-act-pc', options=data['act']['pc'].columns.values, value=data['act']['pc'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-act-pc"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-act-cln', options=data['act']['cln'].columns.values, value=data['act']['cln'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-act-cln', options=data['act']['cln'].columns.values, value=data['act']['cln'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-act-cln', options=data['act']['cln'].columns.values, value=data['act']['cln'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-act-cln"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-acx-pc', options=data['acx']['pc'].columns.values, value=data['acx']['pc'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-acx-pc', options=data['acx']['pc'].columns.values, value=data['acx']['pc'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-acx-pc', options=data['acx']['pc'].columns.values, value=data['acx']['pc'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-acx-pc"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-acx-cln', options=data['acx']['cln'].columns.values, value=data['acx']['cln'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-acx-cln', options=data['acx']['cln'].columns.values, value=data['acx']['cln'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-acx-cln', options=data['acx']['cln'].columns.values, value=data['acx']['cln'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-acx-cln"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-ahx-pc', options=data['ahx']['pc'].columns.values, value=data['ahx']['pc'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-ahx-pc', options=data['ahx']['pc'].columns.values, value=data['ahx']['pc'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-ahx-pc', options=data['ahx']['pc'].columns.values, value=data['ahx']['pc'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-ahx-pc"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-ahx-cln', options=data['ahx']['cln'].columns.values, value=data['ahx']['cln'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-ahx-cln', options=data['ahx']['cln'].columns.values, value=data['ahx']['cln'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-ahx-cln', options=data['ahx']['cln'].columns.values, value=data['ahx']['cln'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-ahx-cln"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-ccx-pc', options=data['ccx']['pc'].columns.values, value=data['ccx']['pc'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-ccx-pc', options=data['ccx']['pc'].columns.values, value=data['ccx']['pc'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-ccx-pc', options=data['ccx']['pc'].columns.values, value=data['ccx']['pc'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-ccx-pc"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-ccx-cln', options=data['ccx']['cln'].columns.values, value=data['ccx']['cln'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-ccx-cln', options=data['ccx']['cln'].columns.values, value=data['ccx']['cln'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-ccx-cln', options=data['ccx']['cln'].columns.values, value=data['ccx']['cln'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-ccx-cln"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-chx-pc', options=data['chx']['pc'].columns.values, value=data['chx']['pc'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-chx-pc', options=data['chx']['pc'].columns.values, value=data['chx']['pc'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-chx-pc', options=data['chx']['pc'].columns.values, value=data['chx']['pc'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-chx-pc"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Dropdown(id='x-axis-dropdown-chx-cln', options=data['chx']['cln'].columns.values, value=data['chx']['cln'].columns.values[0], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='y-axis-dropdown-chx-cln', options=data['chx']['cln'].columns.values, value=data['chx']['cln'].columns.values[1], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            dcc.Dropdown(id='z-axis-dropdown-chx-cln', options=data['chx']['cln'].columns.values, value=data['chx']['cln'].columns.values[2], style={'display': 'inline-block', 'font-size': '10px', 'padding-left': 20, 'padding-right': 20}),
                            html.Div(
                                [
                                    dcc.Graph(id="scatter3d-graph-chx-cln"),
                                ],
                            ),
                        ],
                    ),
                    align="center",
                ),
            ],
        ),
    ],
)

def plot_fig(data, title, twidth, t0, dropdown_x, dropdown_y, dropdown_z):
    
    idx = data.index
    mask = (idx > t0) & (idx < t0 + twidth)

    # All points (grey markers)
    figa = px.scatter_3d(
        data,
        x=dropdown_x,
        y=dropdown_y,
        z=dropdown_z,
        opacity=0.33,
    )
    figa.update_traces(
        marker=dict(size=2, color='gray'),
    )

    # Selected points (colored markers)
    figb = px.scatter_3d(
        data[mask],
        x=dropdown_x,
        y=dropdown_y,
        z=dropdown_z,
        color=data[mask].index,
        color_continuous_scale='Blues',
    )
    figb.update_traces(
        marker=dict(size=3),
        # projection=dict(
        #     x=dict(opacity=0.5, scale=0.667, show=True),
        #     y=dict(opacity=0.5, scale=0.667, show=True),
        #     z=dict(opacity=0.5, scale=0.667, show=True),
        # ),
    )

    # Selected points (grey lines)
    figc = px.line_3d(
        data[mask],
        x=dropdown_x,
        y=dropdown_y,
        z=dropdown_z,
    )
    figc.update_traces(line=dict(color='black', width=1))

    # Combine figures
    # https://stackoverflow.com/questions/65124833/plotly-how-to-combine-scatter-and-line-plots-using-plotly-express
    # https://stackoverflow.com/questions/52863305/plotly-scatter3d-how-can-i-force-3d-axes-to-have-the-same-scale-aspect-ratio
    layout = go.Layout(scene=dict(aspectmode='data'))
    fig = go.Figure(data=figa.data + figb.data + figc.data, layout=layout)
    
    go.scatter3d.projection.X.show = True
    go.scatter3d.projection.Y.show = True
    go.scatter3d.projection.Z.show = True

    # Set graph size
    # https://plotly.com/python/setting-graph-size/
    # https://plotly.com/python/3d-axes/

    fig.update_layout(
        title = {
            'text': title,
            'y':0.95,
            'x':0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        scene=dict(
            xaxis_title=str(dropdown_x),
            yaxis_title=str(dropdown_y),
            zaxis_title=str(dropdown_z),
            # xaxis = dict(nticks=11, range=[-5,5],),
            # yaxis = dict(nticks=11, range=[-5,5],),
            # zaxis = dict(nticks=11, range=[-5,5],),
            # xaxis_visible=False,
            # yaxis_visible=False,
            # zaxis_visible=False,
        ),
        autosize=False,
        width=400,
        height=290,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=4
        ),
    )

    return fig
    
# obs-pc
@app.callback(
    Output("scatter3d-graph-obs-pc", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-obs-pc", "value"),
        Input("y-axis-dropdown-obs-pc", "value"),
        Input("z-axis-dropdown-obs-pc", "value"),
    ]
)

def update_fig_obs_pc(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['obs']['pc'], 'obs-pc', twidth, t0, ddx, ddy, ddz)

# obs-cln
@app.callback(
    Output("scatter3d-graph-obs-cln", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-obs-cln", "value"),
        Input("y-axis-dropdown-obs-cln", "value"),
        Input("z-axis-dropdown-obs-cln", "value"),
    ]
)

def update_fig_obs_cln(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['obs']['cln'], 'obs', twidth, t0, ddx, ddy, ddz)

# act-pc
@app.callback(
    Output("scatter3d-graph-act-pc", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-act-pc", "value"),
        Input("y-axis-dropdown-act-pc", "value"),
        Input("z-axis-dropdown-act-pc", "value"),
    ]
)

def update_fig_act_pc(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['act']['pc'], 'act-pc', twidth, t0, ddx, ddy, ddz)

# act-cln
@app.callback(
    Output("scatter3d-graph-act-cln", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-act-cln", "value"),
        Input("y-axis-dropdown-act-cln", "value"),
        Input("z-axis-dropdown-act-cln", "value"),
    ]
)

def update_fig_act_cln(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['act']['cln'], 'act', twidth, t0, ddx, ddy, ddz)

# acx-pc
@app.callback(
    Output("scatter3d-graph-acx-pc", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-acx-pc", "value"),
        Input("y-axis-dropdown-acx-pc", "value"),
        Input("z-axis-dropdown-acx-pc", "value"),
    ]
)

def update_fig_acx_pc(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['acx']['pc'], 'acx-pc,', twidth, t0, ddx, ddy, ddz)

# acx-cln
@app.callback(
    Output("scatter3d-graph-acx-cln", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-acx-cln", "value"),
        Input("y-axis-dropdown-acx-cln", "value"),
        Input("z-axis-dropdown-acx-cln", "value"),
    ]
)

def update_fig_acx_cln(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['acx']['cln'], 'acx', twidth, t0, ddx, ddy, ddz)

# ahx-pc
@app.callback(
    Output("scatter3d-graph-ahx-pc", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-ahx-pc", "value"),
        Input("y-axis-dropdown-ahx-pc", "value"),
        Input("z-axis-dropdown-ahx-pc", "value"),
    ]
)

def update_fig_ahx_pc(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['ahx']['pc'], 'ahx-pc', twidth, t0, ddx, ddy, ddz)

# ahx-cln
@app.callback(
    Output("scatter3d-graph-ahx-cln", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-ahx-cln", "value"),
        Input("y-axis-dropdown-ahx-cln", "value"),
        Input("z-axis-dropdown-ahx-cln", "value"),
    ]
)

def update_fig_ahx_cln(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['ahx']['cln'], 'ahx', twidth, t0, ddx, ddy, ddz)

# ccx-pc
@app.callback(
    Output("scatter3d-graph-ccx-pc", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-ccx-pc", "value"),
        Input("y-axis-dropdown-ccx-pc", "value"),
        Input("z-axis-dropdown-ccx-pc", "value"),
    ]
)

def update_fig_ccx_pc(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['ccx']['pc'], 'ccx-pc', twidth, t0, ddx, ddy, ddz)

# ccx-cln
@app.callback(
    Output("scatter3d-graph-ccx-cln", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-ccx-cln", "value"),
        Input("y-axis-dropdown-ccx-cln", "value"),
        Input("z-axis-dropdown-ccx-cln", "value"),
    ]
)

def update_fig_ccx_cln(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['ccx']['cln'], 'ccx', twidth, t0, ddx, ddy, ddz)

# chx-pc
@app.callback(
    Output("scatter3d-graph-chx-pc", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-chx-pc", "value"),
        Input("y-axis-dropdown-chx-pc", "value"),
        Input("z-axis-dropdown-chx-pc", "value"),
    ]
)

def update_fig_chx_pc(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['chx']['pc'], 'chx-pc', twidth, t0, ddx, ddy, ddz)

# chx-cln
@app.callback(
    Output("scatter3d-graph-chx-cln", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("time-start-slider", "value"),
        Input("x-axis-dropdown-chx-cln", "value"),
        Input("y-axis-dropdown-chx-cln", "value"),
        Input("z-axis-dropdown-chx-cln", "value"),
    ]
)

def update_fig_chx_cln(twidth, t0, ddx, ddy, ddz):
    return plot_fig(data['chx']['cln'], 'chx', twidth, t0, ddx, ddy, ddz)

app.run_server(debug=False)