import dash
from dash import dcc, html
from flask import Flask
from chatbot.models import get_model_metrics
from dashboard_setup import create_dashboard

# Fetch model metrics
metrics = get_model_metrics()

# Prepare data for visualization
models = list(metrics.keys())
f1_scores = [metrics[model]['f1_score'] for model in models]
precisions = [metrics[model]['precision'] for model in models]
recalls = [metrics[model]['recall'] for model in models]

# Define the layout of the dashboard
dashboard_app = create_dashboard(html.Div([
    html.H1("Model Metrics Dashboard"),
    dcc.Graph(
        id='f1-score-bar',
        figure={
            'data': [
                {'x': models, 'y': f1_scores, 'type': 'bar', 'name': 'F1 Score'},
                {'x': models, 'y': precisions, 'type': 'bar', 'name': 'Precision'},
                {'x': models, 'y': recalls, 'type': 'bar', 'name': 'Recall'},
            ],
            'layout': {
                'title': 'Model Metrics',
                'xaxis': {'title': 'Models'},
                'yaxis': {'title': 'Scores'},
            }
        }
    )
]))