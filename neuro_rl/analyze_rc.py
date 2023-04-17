# https://plotly.com/python/3d-scatter-plots/
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Union, Any
from collections import OrderedDict

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

DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/shadowhand_2023_03_11_1279/'

def process_data(dic: str):
    # read csv file        
    raw_data = pd.read_csv(DATA_PATH + dic + '.csv')
    # remove extra rows of zeros
    clean_data = raw_data.loc[~(raw_data==0).all(axis=1)]
    clean_data.columns = clean_data.columns.astype(str)
    return clean_data

def format_df(data: np.array):
    df = pd.DataFrame(data)
    df.columns = df.columns.astype(str)
    return df

@dataclass
class Data:
    raw: pd.DataFrame
    cos: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.cos = pd.DataFrame(sklearn.metrics.pairwise.cosine_distances(self.raw))

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

# obs = Embeddings(
#     data=Data(process_data('obs')),
#     embeddings={
#         'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
#         'mds': MDSEmbedding(MultiDimensionalScalingEmbedding()),
#         'iso': ISOMAPEmbedding(sklearn.manifold.Isomap(n_components=DIMS, n_neighbors=40)),
#         'lle': LLEEmbedding(sklearn.manifold.LocallyLinearEmbedding(n_components=DIMS, n_neighbors=60, method='modified')),
#         'lem': LEMEmbedding(sklearn.manifold.SpectralEmbedding(n_components=DIMS, affinity='nearest_neighbors', n_neighbors=60)),
#         'tsne': TSNEEmbedding(sklearn.manifold.TSNE(n_components=3, metric='euclidean', perplexity=90, random_state=42)),
#         'umap': UMAPEmbedding(umap.UMAP(n_components=DIMS, metric='cosine', n_neighbors=70, random_state=42))
#     }
# )


obs = Embeddings(
    data=Data(process_data('obs')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

act = Embeddings(
    data=Data(process_data('act')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

acx = Embeddings(
    data=Data(process_data('acx')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

ahx = Embeddings(
    data=Data(process_data('ahx')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

ccx = Embeddings(
    data=Data(process_data('ccx')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

chx = Embeddings(
    data=Data(process_data('chx')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)



logging.info('Finished computing obs embedding')

n_steps = obs.data.raw.shape[0]

def generate_dropdown(id: str, value: Any):
    return dcc.Dropdown(
        id = id,
        options = ['0', '1', '2'],
        value = value,
        style = {
            'display': 'inline-block',
            'font-size': '10px',
            'padding-left': 20,
            'padding-right': 20
        }
    )

def generate_graph(id: str):
    return dcc.Graph(
        id = id
    )
    
# https://community.plotly.com/t/dash-bootstrap-components-grid-system-not-working/30957

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


PLOT_IDS = OrderedDict(
    [
        ('obs-pc', obs.embeddings['pca'].x_embd),
        ('obs-raw', obs.data.raw),
        ('act-pc', act.embeddings['pca'].x_embd),
        ('act-raw', act.data.raw),
        ('acx-pc', acx.embeddings['pca'].x_embd),
        ('acx-raw', acx.data.raw),
        ('ahx-pc', ahx.embeddings['pca'].x_embd),
        ('ahx-raw', ahx.data.raw),
        ('ccx-pc', ccx.embeddings['pca'].x_embd),
        ('ccx-raw', ccx.data.raw),
        ('chx-pc', chx.embeddings['pca'].x_embd),
        ('chx-raw', chx.data.raw)
    ]
)

# PLOT_IDS = OrderedDict(
#     [
#         ('obs-pca', obs.embeddings['pca'].x_embd),
#         ('obs-raw1', obs.data.raw),
#         ('obs-iso', obs.embeddings['iso'].x_embd),
#         ('obs-raw2', obs.data.raw),
#         ('obs-lle', obs.embeddings['lle'].x_embd),
#         ('obs-raw3', obs.data.raw),
#         ('obs-lem', obs.embeddings['lem'].x_embd),
#         ('obs-raw4', obs.data.raw),
#         ('obs-tsne', obs.embeddings['tsne'].x_embd),
#         ('obs-raw5', obs.data.raw),
#         ('obs-umap', obs.embeddings['umap'].x_embd),
#         ('obs-raw6', obs.data.raw),
#     ]
# )

NUM_ROWS = 3
NUM_COLS = 4

# Define the slider layout
slider_layout = html.Div(
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
                    ]
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
                    ]
                ),
                align="center",
            ),
        ],
        # Freeze top row with the time range sliders
        # https://community.plotly.com/t/freeze-one-row-of-dash-using-dashbootstrap/71375/9
        className='sticky-top',
        style={'position': 'sticky-top'},
    )
)

# Define the layout as a grid with M rows and N columns
grid_layout = []
idx = 0
for i in range(NUM_ROWS):
    row = []
    for j in range(NUM_COLS):
        col = [
            html.Div(
                [
                    generate_dropdown('ddx' + '-' + str(idx), '0'),
                    generate_dropdown('ddy' + '-' + str(idx), '1'),
                    generate_dropdown('ddz' + '-' + str(idx), '2'),
                ],
            ),
            html.Div(
                [
                    generate_graph('scatter3d-graph' + '-' + str(idx))
                ]
            ),
        ]
        row.append(
            html.Div(
                col,
                className='col',
                style={'textAlign': 'center'}
            )
        ),
        idx += 1
    grid_layout.append(
        html.Div(
            row,
            className='row'
        )
    )

# Combine the grid layout with the rest of the app layout
app.layout = html.Div(
    [
        html.Div(slider_layout, className='slider-layout'),
        html.Div(grid_layout, className='container-fluid')
    ]
)

def plot_scatter3_ti_tf(data, title, twidth, t0, x, y, z):
    
    idx = np.array(range(np.shape(data)[0]))
    mask = (idx > t0) & (idx <= t0 + twidth)

    # All points (grey markers)
    figa = px.scatter_3d(
        data,
        x=x,
        y=y,
        z=z,
        opacity=0.33,
    )
    figa.update_traces(
        marker=dict(size=2, color='gray'),
    )

    # Selected points (colored markers)
    figb = px.scatter_3d(
        data[mask],
        x=x,
        y=y,
        z=z,
        color=idx[mask],
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
        x=x,
        y=y,
        z=z,
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
            xaxis_title=str(x),
            yaxis_title=str(y),
            zaxis_title=str(z),
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

def rangeslider_tocalendar(idx, data, title):
    @app.callback(
        Output('scatter3d-graph' + '-' + str(idx), "figure"),
        [
            Input("time-window-slider", "value"),
            Input("time-start-slider", "value"),
            Input('ddx' + '-' + str(idx), 'value'),
            Input('ddy' + '-' + str(idx), 'value'),
            Input('ddz' + '-' + str(idx), 'value')
        ]
    )

    def repeated_callback(twidth, t0, ddx, ddy, ddz):
        return plot_scatter3_ti_tf(data, title, twidth, t0, ddx, ddy, ddz)

for i, (key, value) in enumerate(PLOT_IDS.items()):
    rangeslider_tocalendar(i, value, key)

app.run_server(debug=False)
# app.run_server(debug=True, use_reloader=False)
