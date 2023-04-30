from dash import dcc
from typing import Any

def generate_dropdown(id: str, value: Any, options: Any):
    return dcc.Dropdown(
        id = id,
        options = options,
        value = value,
        style = {
            'display': 'inline-block',
            'font-size': '10px',
            'height': '25pt',
            'width': '100%',
            'padding-left': 5,
            'padding-right': 5
        }
    )

def generate_graph(id: str):
    return dcc.Graph(
        id = id
    )