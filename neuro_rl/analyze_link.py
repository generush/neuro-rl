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
from dash_iconify import DashIconify

def generate_sidebar():
    menu = [
        {"icon": "icon-1", "title": "title-1", "href": "href-1"},
        {"icon": "icon-2", "title": "title-2", "href": "href-2"},
        {"icon": "icon-3", "title": "title-3", "href": "href-3"},
        {"icon": "icon-4", "title": "title-4", "href": "href-4"},
        {"icon": "icon-5", "title": "title-5", "href": "href-5"},
    ]

    return html.Div(
        [
            dcc.Link(
                [
                    DashIconify(icon=item["icon"]),
                    html.Div(
                        item["title"],
                    ),
                ],
                href=item["href"],
            )
            for item in menu
        ]
    )

# This was helpful
# https://stackoverflow.com/questions/69647738/dash-output-multiple-graph-based-on-users-graph-choice

# https://community.plotly.com/t/dash-bootstrap-components-grid-system-not-working/30957

app = Dash(__name__)

app.layout = generate_sidebar()

# @app.callback(

#     Output("frrrf","figure"),
#     Input("roiffr","value")
# )

app.run_server(debug=False)
# app.run_server(debug=True, use_reloader=False)

