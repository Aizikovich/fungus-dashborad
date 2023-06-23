import os
from html.parser import HTMLParser
import plotly.graph_objects as go
import dash
import pandas as pd
import plotly.express as px
import requests
from dash import html, dcc, Input, Output


def patch_file(file_path: str, content: bytes, extra: dict = None) -> bytes:
    if file_path == 'index.html':
        index_html_content = content.decode('utf8')
        extra_jsons = f'''
        var patched_jsons_content={{
        {','.join(["'/" + k + "':" + v.decode("utf8") + "" for k, v in extra.items()])}
        }};
        '''
        patched_content = index_html_content.replace(
            '<footer>',
            f'''
            <footer>
            <script>
            ''' + extra_jsons + '''
            const origFetch = window.fetch;
            window.fetch = function () {
                const e = arguments[0]
                if (patched_jsons_content.hasOwnProperty(e)) {
                    return Promise.resolve({
                        json: () => Promise.resolve(patched_jsons_content[e]),
                        headers: new Headers({'content-type': 'application/json'}),
                        status: 200,
                    });
                } else {
                    return origFetch.apply(this, arguments)
                }
            }
            </script>
            '''
        ).replace(
            'href="/',
            'href="'
        ).replace(
            'src="/',
            'src="'
        )
        return patched_content.encode('utf8')
    else:
        return content


def write_file(file_path: str, content: bytes, target_dir='target'):
    target_file_path = os.path.join(target_dir, file_path.lstrip('/').split('?')[0])
    target_leaf_dir = os.path.dirname(target_file_path)
    os.makedirs(target_leaf_dir, exist_ok=True)
    with open(target_file_path, 'wb') as f:
        f.write(content)


class ExternalResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []

    def handle_starttag(self, tag, attrs):
        if tag == 'link':
            for k, v in attrs:
                if k == 'href':
                    self.resources.append(v)
        if tag == 'script':
            for k, v in attrs:
                if k == 'src':
                    self.resources.append(v)


def make_static(base_url, target_dir='target'):
    index_html_bytes = requests.get(base_url).content
    json_paths = ['_dash-layout', '_dash-dependencies', ]
    extra_json = {}
    for json_path in json_paths:
        json_content = requests.get(base_url + json_path).content
        extra_json[json_path] = json_content

    patched_bytes = patch_file('index.html', index_html_bytes, extra=extra_json)
    write_file('index.html', patched_bytes, target_dir)
    parser = ExternalResourceParser()
    parser.feed(patched_bytes.decode('utf8'))
    extra_js = [
        '_dash-component-suites/dash/dcc/async-graph.js',
        '_dash-component-suites/dash/dcc/async-plotlyjs.js',
        '_dash-component-suites/dash/dash_table/async-table.js',
        '_dash-component-suites/dash/dash_table/async-highlight.js'
    ]
    for resource_url in parser.resources + extra_js:
        resource_url_full = base_url + resource_url
        print(f'get {resource_url_full}')
        resource_bytes = requests.get(resource_url_full).content
        patched_bytes = patch_file(resource_url, resource_bytes)
        write_file(resource_url, patched_bytes, target_dir)


def main():
    port = 9050

    app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])

    df = pd.read_csv("../data/mushrooms.csv")

    df_group = df.groupby(['cap-shape', 'class'], as_index=False).size()
    fig = px.bar(df_group, x='cap-shape', y='size', color='class', barmode='group', range_color=[0, 1])
    fig.update_layout(
        title='Number of Edible and Poisonous Mushrooms by Cap Shape',
        xaxis_title='Cap Shape',
        yaxis_title='Count',
        legend_title='Class'
    )

    fig1 = fig3 = fig
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
                    dcc.Graph(id="survivors-gender-pie", figure=fig3)
                ])
            ]),
        ],
    )

    @app.callback(
        [Output('age-fare-scatter', 'figure'),
         Output('survivors-gender-pie', 'figure'),
         Output('app-container', 'style')],
        [Input('mode-slider', 'value'),
         Input('theme-slider', 'value')]
    )
    def update_graphs(mode_value, theme_value):
        # Modify the hues used in the graphs for color-blind mode
        if mode_value == 1:
            fig1.update_traces(marker=dict(color='rgba(255, 0, 0, 0.7)'))
            fig3.update_traces(marker=dict(color=['rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)']))

        # Reset the hues to default for regular mode
        else:
            fig1.update_traces(marker=dict(color='rgba(31, 119, 180, 0.5)'))
            fig3.update_traces(marker=dict(color=['rgba(31, 119, 180, 0.5)', 'rgba(255, 127, 14, 0.5)']))

        # Apply light theme with white background
        if theme_value == 0:
            fig1.update_layout(template="plotly")
            fig3.update_layout(template="plotly")
            return fig1, fig3, {'backgroundColor': '#ffffff', 'color': '#002642'}

        # Apply dark theme with black background
        else:
            fig1.update_layout(template="plotly_dark")
            fig3.update_layout(template="plotly_dark")

            return fig1, fig3, {'backgroundColor': '#002642', 'color': '#E5DADA'}

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


if __name__ == '__main__':
    main()
