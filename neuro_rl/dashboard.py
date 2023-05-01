# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import numpy as np
import pandas as pd

from utils.data_processing import process_data_to_pd
from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

######################## Loading all data files ##########################

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/'

# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/shadowhand_2023_03_11_1279/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_00/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_01/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_AGENT_17_44/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data_AnymalTerrain_Flat_t0_t1000/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/'


DIMS = None # 5

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

dt = 0.005


data = process_data_to_pd(DATA_PATH)
data['TIME'] = data.index




# obs = Embeddings(
#     data=Data(process_data(DATA_PATH + '*-OBS' + '.csv')),
#     embeddings={
#         'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
#     },
#     dt=dt
# )

# act = Embeddings(
#     data=Data(process_data(DATA_PATH + '*-ACT' + '.csv')),
#     embeddings={
#         'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
#     },
#     dt=dt
# )

# ahx = Embeddings(
#     data=Data(process_data(DATA_PATH + '*-AHX' + '.csv')),
#     embeddings={
#         'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
#     },
#     dt=dt
# )

# chx = Embeddings(
#     data=Data(process_data(DATA_PATH + '*-CHX' + '.csv')),
#     embeddings={
#         'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
#     },
#     dt=dt
# )

# pd.DataFrame(ahx.embeddings['pca'].embedding.components_).to_csv('ahx_pc_components.csv', index=False)

# logging.info('Finished computing obs embedding')

n_steps = data.shape[0] # ahx.data.raw.compute().shape[0]

# https://community.plotly.com/t/dash-bootstrap-components-grid-system-not-working/30957

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# obs = data.loc[:,data.columns.str.contains('OBS')].compute()
# act = data.loc[:,data.columns.str.contains('ACT')].compute()
# ahx = data.loc[:,data.columns.str.contains('AHX')].compute()
# chx = data.loc[:,data.columns.str.contains('CHX')].compute()

# PLOT_IDS = OrderedDict(
#     [
#         ('obs-raw', obs),
#         # ('obs-pc', obs.embeddings['pca'].x_embd),
#         ('act-raw', act),
#         # ('act-pc', act.embeddings['pca'].x_embd),
#         ('ahx-raw', ahx),
#         # # ('ahx-pc', ahx.embeddings['pca'].x_embd),
#         ('chx-raw', chx),
#         # # ('chx-pc', chx.embeddings['pca'].x_embd)
#     ]
# )

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

NUM_ROWS = 2
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

NUM_PLOTS = 8
PLOT_NAMES = np.zeros((NUM_PLOTS), dtype=object)
PLOT_NAMES[0] = 'OBS-RAW'
PLOT_NAMES[1] = 'ACT-RAW'
PLOT_NAMES[2] = 'AHX-RAW'
PLOT_NAMES[3] = 'CHX-RAW'
PLOT_NAMES[4] = 'OBS-PC'
PLOT_NAMES[5] = 'ACT-PC'
PLOT_NAMES[6] = 'AHX-PC'
PLOT_NAMES[7] = 'CHX-PC'

dd_options = data.columns.values

dd_defaults = np.zeros((NUM_PLOTS,4), dtype=object)
dd_defaults[0] = ['OBS_RAW_000_u', 'OBS_RAW_001_v', 'OBS_RAW_005_r', 'OBS_RAW_011_r_star']
dd_defaults[1] = ['ACT_RAW_000_LF_HAA', 'ACT_RAW_001_LF_HFE', 'ACT_RAW_002_LF_KFE', 'ACT_RAW_002_LF_KFE']
dd_defaults[2] = ['AHX_RAW_000', 'AHX_RAW_001', 'AHX_RAW_002', 'AHX_RAW_002']
dd_defaults[3] = ['CHX_RAW_000', 'CHX_RAW_001', 'CHX_RAW_002', 'CHX_RAW_002']

dd_defaults[4] = ['OBS_PC_000', 'OBS_PC_001', 'OBS_PC_002', 'OBS_TANGLING']
dd_defaults[5] = ['ACT_PC_000', 'ACT_PC_001', 'ACT_PC_002', 'ACT_TANGLING']
dd_defaults[6] = ['AHX_PC_000', 'AHX_PC_001', 'AHX_PC_002', 'AHX_TANGLING']
dd_defaults[7] = ['CHX_PC_000', 'CHX_PC_001', 'CHX_PC_002', 'CHX_TANGLING']

# Define the layout as a grid with M rows and N columns
grid_layout = []
idx = 0
for i in range(NUM_ROWS):
    row = []
    for j in range(NUM_COLS):
        k = i*NUM_COLS + j
        col = [
            html.Div(
                [
                    html.Div(
                        [
                            html.P("x:"),
                            generate_dropdown('ddx' + '-' + str(idx), dd_defaults[k,0], dd_options),
                        ],
                        style=dict(display='flex')
                    ),
                    html.Div(
                        [
                            html.P("y:"),
                            generate_dropdown('ddy' + '-' + str(idx), dd_defaults[k,1], dd_options),
                        ],
                        style=dict(display='flex')
                    ),
                    html.Div(
                        [
                            html.P("z:"),
                            generate_dropdown('ddz' + '-' + str(idx), dd_defaults[k,2], dd_options),
                        ],
                        style=dict(display='flex')
                    ),
                    html.Div(
                        [
                            html.P("c:"),
                            generate_dropdown('ddc' + '-' + str(idx), dd_defaults[k,3], dd_options),
                        ],
                        style=dict(display='flex')
                    ),
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
def single_callback(idx, plot_name, plot_data):
    @app.callback(
        Output('scatter3d-graph' + '-' + str(idx), "figure"),
        [
            Input("time-window-slider", "value"),
            Input("time-start-slider", "value"),
            Input('ddx' + '-' + str(idx), 'value'),
            Input('ddy' + '-' + str(idx), 'value'),
            Input('ddz' + '-' + str(idx), 'value'),
            Input('ddc' + '-' + str(idx), 'value')
        ]
    )

    def repeated_callback(twidth, t0, ddx, ddy, ddz, ddc):
        return plot_scatter3_ti_tf(plot_name, plot_data, twidth, t0, ddx, ddy, ddz, ddc)

for idx, name in enumerate(PLOT_NAMES):
    single_callback(idx, name, data)

app.run_server(debug=False)
# app.run_server(debug=True, use_reloader=False)
