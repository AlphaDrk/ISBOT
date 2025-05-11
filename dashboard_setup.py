import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
from chatbot.models import get_model_metrics

def create_dashboard(flask_app):
    dashboard_app = dash.Dash(
        __name__,
        server=flask_app,
        routes_pathname_prefix='/dashboard/',
        requests_pathname_prefix='/dashboard/',
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css'
        ]
    )

    # Example dataset (replace with actual data)
    data = pd.read_json('data/data.json')

    # Fetch model metrics
    metrics = get_model_metrics()

    # Prepare data for visualization
    models = list(metrics.keys())
    metric_options = list(metrics[models[0]].keys())

    # Define the layout of the dashboard
    dashboard_app.layout = html.Div([
        html.Div([
            html.H1("ISBOT Project Dashboard", className="text-center my-4"),

            # Dataset Overview
            html.Div([
                html.H2("Dataset Overview", className="my-3"),
                html.P(f"Number of Features: {len(data.columns)}", className="lead"),
                html.P(f"Number of Rows: {len(data)}", className="lead"),
                dash_table.DataTable(
                    id='data-table',
                    columns=[{"name": col, "id": col} for col in data.columns],
                    data=data.head(10).to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px', 'border': '1px solid black'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': 'white', 'color': 'black'}
                )
            ], className="container my-4"),

            # Model Performance
            html.Div([
                html.H2("Model Performance", className="my-3"),
                html.Label("Select Metric:", className="form-label"),
                dcc.Dropdown(
                    id='metric-dropdown',
                    options=[{'label': metric.capitalize(), 'value': metric} for metric in metric_options],
                    value='f1_score',
                    clearable=False,
                    className="form-select"
                ),
                dcc.Graph(id='metric-graph', style={'height': '70vh'}, className="my-4")
            ], className="container my-4"),

            # Footer
            html.Footer("Powered by Dash", className="text-center text-muted my-4")
        ], className="container-fluid")
    ])

    # Callback for updating the graph based on selected metric
    @dashboard_app.callback(
        Output('metric-graph', 'figure'),
        [Input('metric-dropdown', 'value')]
    )
    def update_graph(selected_metric):
        y_values = [metrics[model][selected_metric] for model in models]

        return {
            'data': [
                {
                    'x': models,
                    'y': y_values,
                    'type': 'bar',
                    'name': selected_metric.capitalize(),
                    'hoverinfo': 'x+y'
                }
            ],
            'layout': {
                'title': f'{selected_metric.capitalize()} by Model',
                'xaxis': {'title': 'Models'},
                'yaxis': {'title': 'Score'},
                'plot_bgcolor': '#f9f9f9',
                'paper_bgcolor': '#f9f9f9',
                'hovermode': 'closest'
            }
        }

    return dashboard_app
