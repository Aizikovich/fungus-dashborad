import os
import plotly.graph_objects as go
import dash
import pandas as pd
import plotly.express as px
import requests
from dash import html, dcc, Input, Output


def main():
    port = 9050

    app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])

    df = pd.read_csv("../data/mushrooms.csv")

    fig1 = px.bar(df.groupby(['cap-shape', 'class'], as_index=False).size(),
                  x='cap-shape', y='size', color='class', barmode='group')

    app.layout = html.Div(
        id="app-container",
        className="container",
        style={'backgroundColor': '#ffffff'},  # Set initial background color
        children=[
            html.Div(className="row", children=[
                html.Div(className="six columns", children=[
                    html.H1("Mushroom Dashboard", className="dashboard-title"),
                    html.H2("Eran Aizikovich 316531201, Yuval Segal 313354763",
                            className="dashboard-subtitle")
                ]),
                html.Div(className="six columns", children=[
                    html.Div(className="configuration-box", children=[
                        html.H2("Configuration Box"),
                        html.Div(
                            id="slider-container",
                            className="slider-container",
                            children=[
                                dcc.Slider(
                                    id='mode-slider',
                                    min=0,
                                    max=1,
                                    step=1,
                                    marks={0: 'Regular', 1: 'Color-Blind'},
                                    value=0,
                                    className="slider",
                                ),
                                dcc.Slider(
                                    id='theme-slider',
                                    min=0,
                                    max=1,
                                    step=1,
                                    marks={0: 'Light', 1: 'Dark'},
                                    value=0,
                                    className="slider"
                                )
                            ],
                        ),
                        html.Div(
                            className="button-container",
                            children=[
                                html.Button('Save Static', id='save', n_clicks=0, className="btn btn-primary"),
                                html.Span('', id='saved', className="btn-text")
                            ],
                        ),
                    ]),
                ]),
            ]),
            html.Div(className="row", children=[
                html.Div(className="six columns", children=[
                    dcc.Graph(id="age-fare-scatter", figure=fig1)
                ]),
                html.Div(className="six columns", children=[
                    dcc.Graph(id="survivors-gender-pie")
                ])
            ]),
        ],
    )

    @app.callback(
        [Output('age-fare-scatter', 'figure'),
         Output('survivors-gender-pie', 'figure')],
        [Input('mode-slider', 'value'),
         Input('theme-slider', 'value')]
    )
    def update_graphs(mode_value, theme_value):
        # Modify the hues used in the graphs for color-blind mode
        if mode_value == 1:
            fig1.update_traces(marker=dict(color='rgba(255, 0, 0, 0.7)'))
            pie_fig = update_pie_chart(df, 'cap-shape', mode_value)
        else:
            fig1.update_traces(marker=dict(color='rgba(31, 119, 180, 0.5)'))
            pie_fig = update_pie_chart(df, 'cap-shape', mode_value)

        # Apply light theme with white background
        if theme_value == 0:
            fig1.update_layout(template="plotly")
            pie_fig.update_layout(template="plotly")
            return fig1, pie_fig

        # Apply dark theme with black background
        else:
            fig1.update_layout(template="plotly_dark")
            pie_fig.update_layout(template="plotly_dark")
            return fig1, pie_fig

    @app.callback(
        Output('saved', 'children'),
        Input('save', 'n_clicks')
    )
    def save_result(n_clicks):
        if n_clicks == 0:
            return 'Not saved'
        else:
            make_static(f'http://127.0.0.1:{port}/')
            return 'Saved'

    app.run_server(debug=False, port=port)


def update_pie_chart(data, attribute, mode_value):
    df_group = data.groupby([attribute, 'class'], as_index=False).size()
    if mode_value == 1:
        fig = px.pie(df_group, values='size', names=attribute, color='class',
                     title=f"Number of Edible and Poisonous Mushrooms by {attribute} (Color-Blind Mode)",
                     template="plotly")
    else:
        fig = px.pie(df_group, values='size', names=attribute, color='class',
                     title=f"Number of Edible and Poisonous Mushrooms by {attribute}",
                     template="plotly")
    return fig


if __name__ == '__main__':
    main()
