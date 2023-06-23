import os
from html.parser import HTMLParser
import random

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


most_informative_attributes = {'odor': 'f', 'spore-print-color': 'n', 'gill-color': 'b', 'ring-type': 'p'}


def update_pie_chart(dt, attribute, attribute_value):
    data = dt.copy()
    attribute_mapping = {
        'cap-shape': {
            'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'
        },
        'cap-surface': {
            'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'
        },
        'cap-color': {
            'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green',
            'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'
        },
        'bruises': {
            't': 'bruises', 'f': 'no bruises'
        },
        'odor': {
            'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul',
            'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'
        },
        'gill-attachment': {
            'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'
        },
        'gill-spacing': {
            'c': 'close', 'w': 'crowded', 'd': 'distant'
        },
        'gill-size': {
            'b': 'broad', 'n': 'narrow'
        },
        'gill-color': {
            'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray',
            'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red',
            'w': 'white', 'y': 'yellow'
        },
        'stalk-shape': {
            'e': 'enlarging', 't': 'tapering'
        },
        'stalk-root': {
            'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs',
            'r': 'rooted', '?': 'missing'
        },
        'stalk-surface-above-ring': {
            'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'
        },
        'stalk-surface-below-ring': {
            'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'
        },
        'stalk-color-above-ring': {
            'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange',
            'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'
        },
        'stalk-color-below-ring': {
            'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange',
            'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'
        },
        'veil-type': {
            'p': 'partial', 'u': 'universal'
        },
        'veil-color': {
            'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'
        },
        'ring-number': {
            'n': 'none', 'o': 'one', 't': 'two'
        },
        'ring-type': {
            'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large',
            'n': 'none', 'p': 'pendant', 's': 'sheathing', 'z': 'zone'
        },
        'spore-print-color': {
            'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green',
            'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'
        },
        'population': {
            'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered',
            'v': 'several', 'y': 'solitary'
        },
        'habitat': {
            'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths',
            'u': 'urban', 'w': 'waste', 'd': 'woods'
        },
        'class': {
            'e': 'edible',
            'p': 'poisonous'
        }
    }

    # Apply attribute mapping
    if attribute in attribute_mapping:
        data[attribute] = data[attribute].map(attribute_mapping[attribute])
    data['class'] = data['class'].map(attribute_mapping['class'])

    # Filter data based on attribute value
    filtered_data = data[data[attribute] == attribute_mapping[attribute][attribute_value]]

    # Check if filtered_data is empty
    if filtered_data.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No Data Available")
        return empty_fig

    # Select the 'class' and filtered attribute columns
    filtered_data = filtered_data[['class', attribute]]

    # Group the filtered data by class
    df_group = filtered_data.groupby('class').size().reset_index(name='count')

    print(filtered_data)

    fig = px.pie(df_group, values='count', names='class',
                 title=f"Proportions of Edible and Poisonous Mushrooms with '{attribute}' = "
                       f"'{attribute_mapping[attribute][attribute_value]}'",
                 template="plotly")
    fig.update_traces(marker=dict(line=dict(color='#000000', width=0.5)))
    fig.update_coloraxes(colorbar=dict(outlinecolor='#000000', outlinewidth=0.5))
    return fig


def main():
    port = 9050

    app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])

    df = pd.read_csv("../data/mushrooms.csv")
    attributes = df.columns.tolist()  # Get the list of attribute options from the DataFrame columns

    fig = update_pie_chart(df, 'habitat', 'g')  # Initialize the pie chart with a default attribute and mode

    df_group = df.groupby(['cap-surface', 'class'], as_index=False).size()
    fig1 = px.bar(df_group, x='cap-surface', y='size', color='class', barmode='group', range_color=[0, 1])
    fig1.update_layout(
        title='Number of Edible and Poisonous Mushrooms by Cap Surface',
        xaxis_title='Cap Surface',
        yaxis_title='Count',
        legend_title='Class'
    )
    app.layout = html.Div(
        id="app-container",
        className="container",
        style={'backgroundColor': '#ffffff'},  # Set initial background color
        children=[
            html.Div(className="row", children=[
                html.Div(className="six columns", children=[
                    html.H1("Mushroom Dashboard", className="dashboard-title"),
                    html.H2("Eran Aizikovich 316531201, Yuval Segal 313354763",
                            className="dashboard-subtitle"),

                ]),
                html.Div(className="charts-container", children=[
                    html.Div(className="chart-column", children=[
                        dcc.Graph(id="age-fare-scatter", figure=fig1)
                    ]),
                    html.Div(className="chart-column", children=[
                        html.Div(className="pie-chat", children=[
                            dcc.Graph(id="survivors-gender-pie", figure=fig),
                            html.Div(className="attribute-dropdown", children=[
                                html.Label("Attribute for Pie Chart", style={'font-family': "Arial Black"}),
                                dcc.Dropdown(
                                    id='attribute-dropdown',
                                    options=[{'label': attr, 'value': attr} for attr in attributes[1:]],
                                    value=attributes[1],
                                    multi=False,
                                    optionHeight=40,  # Set the height of each option

                                    style={'height': 'auto', 'width': '100%', 'font-family': "Arial Black",
                                           'color': '#840032'},
                                ),
                            ])
                        ]),
                    ]),
                ]),
            ]),
            html.Div(className="row", children=[
                html.Div(className="twelve columns", children=[
                    html.Div(className="configuration-box", children=[
                        html.H2("Configuration Box"),
                        html.Div(id="slider-container", className="slider-container", children=[
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
                            ),
                        ]),
                        html.Div(className="button-container", children=[
                            html.Button('Save Static', id='save', n_clicks=0, className="btn btn-primary"),
                            html.Span('', id='saved', className="btn-text")
                        ]),
                    ]),
                ]),
            ]),
        ],
    )

    @app.callback(
        [Output('age-fare-scatter', 'figure'),
         Output('survivors-gender-pie', 'figure'),
         Output('app-container', 'style')],
        [Input('mode-slider', 'value'),
         Input('theme-slider', 'value'),
         Input('attribute-dropdown', 'value')]
    )
    def update_graphs(mode_value, theme_value, attribute_value):
        attribute_values = df[
            attribute_value].unique().tolist()  # Get the unique attribute values for the selected attribute
        chosen_value = random.choice(attribute_values)
        print(attribute_value, chosen_value)

        fig = update_pie_chart(df, attribute_value, chosen_value)

        # Modify the hues used in the graphs for color-blind mode
        if mode_value == 1:
            fig1.update_traces(marker=dict(color='rgba(255, 0, 0, 0.7)'))
            fig.update_traces(marker=dict(colors=['rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)']))

        # Reset the hues to default for regular mode
        else:
            fig1.update_traces(marker=dict(color='rgba(31, 119, 180, 0.5)'))
            fig.update_traces(marker=dict(colors=['rgba(31, 119, 180, 0.5)', 'rgba(255, 127, 14, 0.5)']))

        # Apply light theme with white background
        if theme_value == 0:
            fig1.update_layout(template="plotly")
            fig.update_layout(template="plotly")
            return fig1, fig, {'backgroundColor': '#ffffff', 'color': '#002642'}

        # Apply dark theme with black background
        else:
            fig1.update_layout(template="plotly_dark")
            fig.update_layout(template="plotly_dark")
            fig.update_traces(textfont_color='#FFFFFF')

            return fig1, fig, {'backgroundColor': '#002642', 'color': '#E5DADA'}

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