# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from utils.data_processing import process_data
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


obs = Embeddings(
    data=Data(process_data(DATA_PATH + '*-OBS' + '.parquet')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

act = Embeddings(
    data=Data(process_data(DATA_PATH + '*-ACT' + '.parquet')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

acx = Embeddings(
    data=Data(process_data(DATA_PATH + '*-ACX' + '.parquet')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

ahx = Embeddings(
    data=Data(process_data(DATA_PATH + '*-AHX' + '.parquet')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

ccx = Embeddings(
    data=Data(process_data(DATA_PATH + '*-CCX' + '.parquet')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)

chx = Embeddings(
    data=Data(process_data(DATA_PATH + '*-CHX' + '.parquet')),
    embeddings={
        'pca': PCAEmbedding(sklearn.decomposition.PCA(n_components=DIMS)),
    }
)



logging.info('Finished computing obs embedding')

n_steps = acx.data.raw.compute().shape[0]

# https://community.plotly.com/t/dash-bootstrap-components-grid-system-not-working/30957

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


PLOT_IDS = OrderedDict(
    [
        ('obs-pc', obs.embeddings['pca'].x_embd),
        # ('obs-raw', obs.data.raw.compute()),
        ('act-pc', act.embeddings['pca'].x_embd),
        # ('act-raw', act.data.raw.compute()),
        ('acx-pc', acx.embeddings['pca'].x_embd),
        # ('acx-raw', acx.data.raw.compute()),
        ('ahx-pc', ahx.embeddings['pca'].x_embd),
        # ('ahx-raw', ahx.data.raw.compute()),
        ('ccx-pc', ccx.embeddings['pca'].x_embd),
        # ('ccx-raw', ccx.data.raw.compute()),
        ('chx-pc', chx.embeddings['pca'].x_embd),
        # ('chx-raw', chx.data.raw.compute())
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
