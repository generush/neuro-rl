import dash
import dash_html_components as html

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

M = 3
N = 4

# Define the layout as a grid with M rows and N columns
grid_layout = []
for i in range(M):
    row = []
    for j in range(N):
        col = html.Div([
            html.H1(f"Row {i + 1}, Column {j + 1}")
        ], className='col')
        row.append(col)
    grid_layout.append(html.Div(row, className='row'))

# Combine the grid layout with the rest of the app layout
app.layout = html.Div([
    html.H1("My Dashboard"),
    html.Div(grid_layout, className='container-fluid')
])

if __name__ == '__main__':
    app.run_server(debug=False)
