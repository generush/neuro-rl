
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_scatter3_ti_tf(title, data, t0, twidth, x, y, z, c, folder_path: str, file_suffix: str=''):
    
    # TODO TO DO: remove hard coding of 1/dt here
    t = np.array(data['TIME'])
    mask = (t > t0) & (t <= t0 + twidth)

    # All points (grey markers)
    # figa = px.scatter_3d(
    #     data,
    #     x=x,
    #     y=y,
    #     z=z,
    #     opacity=0.33,
    # )
    # figa.update_traces(
    #     marker=dict(size=2, color='gray'),
    # )

    # Selected points (colored markers)
    figb = px.scatter_3d(
        data[mask],
        x=x,
        y=y,
        z=z,
        color=c,
        # color_continuous_scale='Blues',
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
    fig = go.Figure(data=figb.data + figc.data, layout=layout)
    # fig = go.Figure(data=figb.data, layout=layout)

    
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
        legend_title=str(c),
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

    fig.write_image(folder_path + '/' + title + file_suffix + '.svg')

    return fig
