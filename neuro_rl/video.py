from dash import Dash, dcc, html, Input, Output, State
import dash_player

PATH = 'assets/rl-video-step-0.mov'

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    style={"width": "48%", "padding": "0px"},
                    children=[
                        dash_player.DashPlayer(
                            id="player",
                            url=PATH,
                            controls=True,
                            # width="100%",
                            # height="250px",
                            intervalCurrentTime=50
                        ),
                        html.Div(
                            [
                                html.Div(
                                    id="current-time-div",
                                    style={"margin": "10px 0px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flex-direction": "column",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "48%", "padding": "10px"},
                    children=[
                        html.P("Volume:", style={"marginTop": "30px"}),
                        dcc.Slider(
                            id="volume-slider",
                            min=0,
                            max=100,
                            step=0.1,
                            value=50,
                            updatemode="drag",
                        ),
                    ],
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "space-between",
            },
        ),
    ]
)

@app.callback(
    Output("current-time-div", "children"),
    Input("player", "currentTime"),
)
def display_currentTime(currentTime):
    return f"Current Time: {currentTime}"

@app.callback(
    Output("player", "seekTo"),
    Input("volume-slider", "value"),
)
def set_volume(value):
    return value

if __name__ == "__main__":
    app.run_server(debug=False)