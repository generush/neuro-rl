# https://plotly.com/python/3d-scatter-plots/
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Union, Any

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
from dash_iconify import DashIconify


n_steps = 100


d = pd.DataFrame(
    {
        1: [45,4,67],
        2: [4,4,67],
        3: [45,74,67],
        4: [45,4,607],
        5: [450,40,67],
        6: [-45,4,67],
        7: [-45,4,-67],
        8: [45,4,-67],
        9: [45,-4,-67],
        10: [-45,4,-67],
        11: [485,-4,67],
        12: [45,-4,967],
    }
)

def generate_dropdown(id: str, value: Any):
    return dcc.Dropdown(
        id = id,
        options = [1,2,3],
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

PLOT_IDS = {}
    'obs-pc',
    'obs-raw',
    'act-pc',
    'act-raw',
    'acx-pc',
    'acx-raw',
    'ahx-pc',
    'ahx-raw',
    'ccx-pc',
    'ccx-raw',
    'chx-pc',
    'chx-raw'
]

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
                            max=50, #n_steps,
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
                    generate_dropdown('ddx' + '-' + PLOT_IDS[idx], 1),
                    generate_dropdown('ddy' + '-' + PLOT_IDS[idx], 2),
                    generate_dropdown('ddz' + '-' + PLOT_IDS[idx], 3),
                ],
            ),
            html.Div(
                [
                    generate_graph('scatter3d-graph' + '-' + PLOT_IDS[idx])
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

def rangeslider_tocalendar(title):
    @app.callback(
        Output('scatter3d-graph' + '-' + title, "figure"),
        [
            Input("time-window-slider", "value"),
            Input("time-start-slider", "value"),
            Input('ddx' + '-' + title, 'value'),
            Input('ddy' + '-' + title, 'value'),
            Input('ddz' + '-' + title, 'value')
        ]
    )

    def repeated_callback(twidth, t0, ddx, ddy, ddz):
        return plot_scatter3_ti_tf(d, title, twidth, t0, ddx, ddy, ddz)

for k in range(NUM_ROWS * NUM_COLS):
    rangeslider_tocalendar(PLOT_IDS[k])

app.run_server(debug=False)
# app.run_server(debug=True, use_reloader=False)
