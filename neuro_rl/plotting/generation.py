from dash import dcc
from typing import Any

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