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

def run_dashboard(path: str, file_suffix: str = ''):

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

    data = process_data_to_pd(path, 'DATA' + file_suffix)

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

    t0 = data['TIME'].min()
    tf = data['TIME'].max()
    dt = data['TIME'][1] - data['TIME'][0]
    # n_steps = data.shape[0] # ahx.data.raw.compute().shape[0]

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
                            html.P("Time start:"),
                            dcc.Slider(
                                id='time-start-slider',
                                min=t0,
                                max=tf,
                                step=dt,
                                value=t0,
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
                            html.P("Time window width:"),
                            dcc.Slider(
                                id='time-window-slider',
                                min=dt,
                                max=tf - t0,
                                step=dt,
                                value=tf - t0,
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
    PLOT_NAMES[0] = 'Actions raw'
    PLOT_NAMES[1] = 'Observations'
    PLOT_NAMES[2] = 'A_MLP_XX'
    PLOT_NAMES[3] = 'A_LSTM_CX'
    PLOT_NAMES[4] = 'A_LSTM_C1X'
    PLOT_NAMES[5] = 'A_LSTM_C2X'
    PLOT_NAMES[6] = 'A_LSTM_HX'
    PLOT_NAMES[7] = 'Actions'
    # PLOT_NAMES[8] = 'AGRU_HX-PC'
    # PLOT_NAMES[9] = 'CGRU_HX-PC'


    dd_options = data.columns.values

    dd_defaults = np.zeros((NUM_PLOTS,4), dtype=object)




    dd_defaults[0] = ['ACT_PC_000', 'ACT_PC_001', 'ACT_PC_002', 'TIME']
    dd_defaults[1] = ['ACT_PC_000', 'ACT_PC_001', 'ACT_PC_002', 'ENV']
    dd_defaults[2] = ['ACT_PC_000', 'ACT_PC_001', 'ACT_PC_002', 'ACT_PC_002']
    dd_defaults[3] = ['ACT_PC_000', 'ACT_PC_001', 'ACT_PC_002', 'ACT_PC_002']
    dd_defaults[4] = ['A_LSTM_HC_PC_000', 'A_LSTM_HC_PC_001', 'A_LSTM_HC_PC_002', 'TIME']
    dd_defaults[5] = ['A_LSTM_HC_PC_000', 'A_LSTM_HC_PC_001', 'A_LSTM_HC_PC_002', 'ENV']
    dd_defaults[6] = ['A_LSTM_HC_PC_000', 'A_LSTM_HC_PC_001', 'A_LSTM_HC_PC_002', 'A_LSTM_HC_PC_002']
    dd_defaults[7] = ['A_LSTM_HC_PC_000', 'A_LSTM_HC_PC_001', 'A_LSTM_HC_PC_002', 'A_LSTM_HC_PC_002']








    # # dd_defaults[0] = ['OBS_RAW_018_obj_qx', 'OBS_RAW_019_obj_qy', 'OBS_RAW_020_obj_qz', 'CONDITION']
    # # dd_defaults[1] = ['ACT_RAW_000_WRJ1', 'ACT_RAW_001_WRJ0', 'ACT_RAW_002_FFJ3', 'CONDITION']
    # # dd_defaults[2] = ['AHX_RAW_000', 'AHX_RAW_001', 'AHX_RAW_002', 'CONDITION']
    # # dd_defaults[3] = ['CHX_RAW_000', 'CHX_RAW_001', 'CHX_RAW_002', 'CONDITION']

    # # dd_defaults[4] = ['OBS_PC_000', 'OBS_PC_001', 'OBS_PC_002', 'CONDITION']
    # # dd_defaults[5] = ['ACT_PC_000', 'ACT_PC_001', 'ACT_PC_002', 'CONDITION']
    # # dd_defaults[6] = ['AHX_PC_000', 'AHX_PC_001', 'AHX_PC_002', 'CONDITION']
    # # dd_defaults[7] = ['CHX_PC_000', 'CHX_PC_001', 'CHX_PC_002', 'CONDITION']

    # # dd_defaults[0] = ['OBS_RAW_000_u', 'OBS_RAW_001_v', 'OBS_RAW_005_r', 'OBS_RAW_011_r_star']
    # dd_defaults[0] = ['ACT_RAW_000_LF_HAA', 'ACT_RAW_001_LF_HFE', 'ACT_RAW_002_LF_KFE', 'ACT_RAW_002_LF_KFE']

    # dd_defaults[1] = ['OBS_PC_000', 'OBS_PC_001', 'OBS_PC_002', 'ENV']
    # dd_defaults[2] = ['A_MLP_XX_PC_000', 'A_MLP_XX_PC_001', 'A_MLP_XX_PC_002', 'ENV']
    # dd_defaults[3] = ['A_LSTM_CX_PC_000', 'A_LSTM_CX_PC_001', 'A_LSTM_CX_PC_002', 'ENV']
    # dd_defaults[4] = ['A_LSTM_C1X_PC_000', 'A_LSTM_C1X_PC_001', 'A_LSTM_C1X_PC_002', 'ENV']
    # dd_defaults[5] = ['A_LSTM_C2X_PC_000', 'A_LSTM_C2X_PC_001', 'A_LSTM_C2X_PC_002', 'ENV']
    # dd_defaults[6] = ['A_LSTM_HX_PC_000', 'A_LSTM_HX_PC_001', 'A_LSTM_HX_PC_002', 'ENV']
    # dd_defaults[7] = ['ACT_PC_000', 'ACT_PC_001', 'ACT_PC_002', 'ENV']

    # # dd_defaults[2] = ['OBS_PC_000', 'OBS_PC_001', 'OBS_PC_002', 'OBS_TANGLING']
    # # dd_defaults[3] = ['ACT_PC_000', 'ACT_PC_001', 'ACT_PC_002', 'ACT_TANGLING']
    # # dd_defaults[4] = ['ALSTM_HX_PC_000', 'ALSTM_HX_PC_001', 'ALSTM_HX_PC_002', 'ALSTM_HX_TANGLING']
    # # dd_defaults[5] = ['ALSTM_CX_PC_000', 'ALSTM_CX_PC_001', 'ALSTM_CX_PC_002', 'ALSTM_CX_TANGLING']
    # # dd_defaults[6] = ['CLSTM_HX_PC_000', 'CLSTM_HX_PC_001', 'CLSTM_HX_PC_002', 'CLSTM_HX_TANGLING']
    # # dd_defaults[7] = ['CLSTM_CX_PC_000', 'CLSTM_CX_PC_001', 'CLSTM_CX_PC_002', 'CLSTM_CX_TANGLING']

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
                Input("time-start-slider", "value"),
                Input("time-window-slider", "value"),
                Input('ddx' + '-' + str(idx), 'value'),
                Input('ddy' + '-' + str(idx), 'value'),
                Input('ddz' + '-' + str(idx), 'value'),
                Input('ddc' + '-' + str(idx), 'value')
            ]
        )

        def repeated_callback(t0, twidth, ddx, ddy, ddz, ddc):
            return plot_scatter3_ti_tf(plot_name, plot_data, t0, twidth, ddx, ddy, ddz, ddc, path, file_suffix)

    for idx, name in enumerate(PLOT_NAMES):
        single_callback(idx, name, data)

    app.run_server(debug=False)
    # app.run_server(debug=True, use_reloader=False)
