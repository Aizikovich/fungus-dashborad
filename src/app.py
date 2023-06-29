import base64
import math
import os
from collections import Counter
from html.parser import HTMLParser
import scipy.stats as ss

import numpy as np
import plotly.graph_objects as go
import dash
import pandas as pd
import plotly.express as px
import requests
from dash import html, dcc, Input, Output

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
server = app.server

color_blind_list = [
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#1f77b4',  # Blue
    '#8c564b',  # Brown
    '#9467bd',  # Purple
    '#d62728',  # Red
    '#ffbb78',  # Light Orange
    '#aec7e8',  # Light Blue
    '#17becf',  # Teal
    '#ff9896',  # Light Red
    '#c49c94',  # Light Brown
    '#e377c2',  # Pink
    '#98df8a',  # Light Green
    '#ff7f0e',  # Orange
    '#dbdb8d',  # Light Olive
    '#f7b6d2',  # Light Pink
    '#2ca02c',  # Green
    '#9edae5',  # Light Teal
    '#c7c7c7',  # Light Gray
    '#c5b0d5',  # Light Purple
]

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


# def update_bar_plot(dt, attribute, mode_value):
#     df = dt.copy()
#
#     # Apply attribute mapping
#     if attribute in attribute_mapping:
#         df[attribute] = df[attribute].map(attribute_mapping[attribute])
#     df['class'] = df['class'].map(attribute_mapping['class'])
#     #
#     df_group = df.groupby([attribute, 'class'], as_index=False).size()
#
#     if mode_value == 1:
#         colors = ['rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)']  # Color-blind mode colors
#     else:
#         colors = ['rgba(31, 119, 180, 0.5)', 'rgba(255, 127, 14, 0.5)']  # Default colors
#
#     fig1 = px.bar(df_group, x=attribute, y='size', color='class', barmode='group', range_color=[0, 1],
#                   color_discrete_sequence=colors)
#
#     fig1.update_layout(
#         title=f'Number of Edible and Poisonous Mushrooms by {attribute}',
#         xaxis_title=attribute,
#         yaxis_title='Count',
#         legend_title='Class'
#     )
#     return fig1


# def update_bar_plot(dt, attribute, mode_value):
#     df = dt.copy()
#
#     # Apply attribute mapping
#     if attribute in attribute_mapping:
#         df[attribute] = df[attribute].map(attribute_mapping[attribute])
#     df['habitat'] = df['habitat'].map(attribute_mapping['habitat'])
#     #
#     df_group = df.groupby([attribute, 'habitat'], as_index=False).size()
#
#     if mode_value == 1:
#         colors = ['rgba(255, 0, 0, 0.5)', 'rgba(0, 0, 255, 0.5)']  # Color-blind mode colors
#     else:
#         colors = ['rgba(31, 119, 180, 0.5)', 'rgba(255, 127, 14, 0.5)']  # Default colors
#
#     fig1 = px.bar(df_group, x=attribute, y='size', color='habitat', barmode='group', range_color=[0, 1])
#                   # color_discrete_sequence=colors)
#
#     fig1.update_layout(
#         title=f'Number of Edible and Poisonous Mushrooms by {attribute}',
#         xaxis_title=attribute,
#         yaxis_title='Count',
#         legend_title='habitat'
#     )
#     return fig1

def update_bar_plot(dt, attribute, color_blind_mode):
    df = dt.copy()

    # Apply attribute mapping
    if attribute in attribute_mapping:
        df[attribute] = df[attribute].map(attribute_mapping[attribute])
    df['habitat'] = df['habitat'].map(attribute_mapping['habitat'])

    # Create a new DataFrame to hold the unique combinations of habitat and medal
    df_normalized = df.groupby(['habitat', attribute]).size().reset_index(name='count')

    # Set the sum value to 1 for habitat
    df_normalized['count'] = df_normalized.groupby('habitat')['count'].apply(lambda x: x / x.sum())

    if color_blind_mode == 1:
        colors = color_blind_list
    else:
        colors = px.colors.qualitative.Plotly

    fig = px.bar(df_normalized, x="habitat", y="count", color=attribute,
                 pattern_shape=attribute, pattern_shape_sequence=['', '/', '\\', 'x', '-', '|', '+', '.'],
                 color_discrete_sequence=colors)

    return fig


def update_pie_chart(dt, attribute):
    data = dt.copy()

    # Apply attribute mapping
    if attribute in attribute_mapping:
        data[attribute] = data[attribute].map(attribute_mapping[attribute])
    data['class'] = data['class'].map(attribute_mapping['class'])

    # Select the 'class' and filtered attribute columns
    filtered_data = data[[attribute]]

    # Group the filtered data by class
    df_group = filtered_data.groupby(attribute).size().reset_index(name='count')

    fig = px.pie(df_group, values='count', names=attribute,
                 title=f"{attribute} Distribution ",
                 template="plotly")
    fig.update_traces(marker=dict(line=dict(color='#000000', width=0.5)))
    fig.update_coloraxes(colorbar=dict(outlinecolor='#000000', outlinewidth=0.5))
    return fig


# Calculate conditional entropy
def conditional_entropy(x, y):
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


# Calculate Theil's U
def theil_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


port = 9050

df = pd.read_csv("../data/mushrooms.csv")
df = df.drop(['veil-type'], axis=1)

attributes = df.columns.tolist()  # Get the list of attribute options from the DataFrame columns

fig = update_pie_chart(df, 'habitat')  # Initialize the pie chart with a default attribute and mode

fig1 = update_bar_plot(df, 'ring-number', 0)

# Map the attribute values to their corresponding labels
df_combinations = df.copy()
df_combinations["class"] = df_combinations["class"].map(attribute_mapping["class"])
df_combinations["bruises"] = df_combinations["bruises"].map(attribute_mapping["bruises"])
df_combinations["population"] = df_combinations["population"].map(attribute_mapping["population"])
df_combinations["habitat"] = df_combinations["habitat"].map(attribute_mapping["habitat"])
df_combinations["odor"] = df_combinations["odor"].map(attribute_mapping["odor"])
df_combinations["spore-print-color"] = df_combinations["spore-print-color"].map(
    attribute_mapping["spore-print-color"])
df_combinations["gill-color"] = df_combinations["gill-color"].map(attribute_mapping["gill-color"])
df_combinations["stalk-shape"] = df_combinations["stalk-shape"].map(attribute_mapping["stalk-shape"])
df_combinations["gill-size"] = df_combinations["gill-size"].map(attribute_mapping["gill-size"])
df_combinations["ring-number"] = df_combinations["ring-number"].map(attribute_mapping["ring-number"])

# filter records that odor label is appears less than 50 times
df_odor = df_combinations.groupby('odor').filter(lambda x: len(x) > 300)
df_spore_color = df_odor.groupby(['odor', 'spore-print-color']).filter(lambda x: len(x) > 300)
df_gill_color = df_spore_color.groupby(['odor', 'spore-print-color', 'gill-color']).filter(lambda x: len(x) > 300)


def create_heatmap(color_blind):
    # Calculate Theil's U for each column
    theilu = pd.DataFrame(index=['class'], columns=df.columns)
    for column in df.columns:
        u = theil_u(df['class'].tolist(), df[column].tolist())
        theilu.loc[:, column] = u

    # Remove columns with values less than 0.4 and higher than 0.1
    theilu = theilu.loc[:, (theilu.min() >= 0.3)]

    # Remove the 'class' attribute
    if 'class' in theilu.columns:
        theilu = theilu.drop('class', axis=1)

    # Sort values from high to low
    theilu = theilu.sort_values(by='class', axis=1, ascending=False)

    if color_blind == 1:
        colorscale = 'Cividis'
    else:
        colorscale = 'Viridis'
    theilu_fig = go.Figure(
        data=[
            go.Heatmap(
                z=theilu.values,
                x=theilu.columns,
                y=theilu.index,
                colorscale=colorscale,
                colorbar=dict(title="Theil's U"),
            )
        ],
        layout=go.Layout(
            title="Theil's U Heatmap",
            xaxis=dict(title="Features"),
            yaxis=dict(title="Class"),
            height=500,
        ),
    )
    return theilu_fig


theilu_fig = create_heatmap(0)


# Create the sunburst chart
def create_sunburst(color_blind):
    if color_blind:
        color_discrete_map = {'edible': '#1f77b4', 'poisonous': '#ff7f0e'}
    else:
        color_discrete_map = {'edible': 'seagreen', 'poisonous': 'red'}

    fi = px.sunburst(
        df_gill_color,
        path=["odor", "spore-print-color", "gill-color", "class"],
        color="class",
        color_discrete_map=color_discrete_map,
        height=800,
    )
    fi.update_traces(textinfo='label+percent entry')
    fi.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fi


fig3 = create_sunburst(color_blind=False)


def create_big_heatmap(color_blind):
    # Calculate the chi-square test p-values for variable association
    def cramers_v(confusion_matrix):
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            confusion_matrix = pd.crosstab(df[col1], df[col2])
            corr_matrix.loc[col1, col2] = cramers_v(confusion_matrix)

    # Set diagonal elements to 1
    np.fill_diagonal(corr_matrix.values, 1)
    # Plot the heatmap
    if color_blind:
        scale = 'Cividis'
    else:
        scale = 'Viridis'
    c_fig = px.imshow(corr_matrix.astype(float),
                      x=corr_matrix.columns,
                      y=corr_matrix.index,
                      color_continuous_scale=scale,
                      title="Cramér's V Statistic Heatmap",
                      height=600,
                      width=600)
    return c_fig


corr_fig = create_big_heatmap(color_blind=0)
encoded_image = base64.b64encode(open("img.png", 'rb').read())

app.layout = html.Div(
    id="app-container",
    className="container",
    style={'backgroundColor': '#ffffff'},  # Set initial background color
    children=[

        html.Div(className="six columns", children=[
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), className="logo"),

            html.H1("Mushroom Dashboard", className="dashboard-title"),
            html.H2("Eran Aizikovich 316531201, Yuval Segal 313354763",
                    className="dashboard-subtitle"),

        ]),

        html.Div(className="charts-container", children=[
            html.Div(className="row", children=[
                html.Div(className="chart-column", children=[
                    html.Div(className="corr_heatmap", children=[
                        dcc.Graph(
                            id='theil-u-heatmap',
                            figure=theilu_fig,
                        ),
                    ]),
                ]),
                html.Div(className="chart-column", children=[
                    html.Div(className="corr_heatmap", children=[
                        dcc.Graph(
                            id='corr_heatmap',
                            figure=corr_fig,
                        ),
                    ], style={'margin': 'auto'}),
                ]),
            ]),

            html.Div(className="row", children=[

                html.Div(className="chart-column", children=[
                    dcc.Graph(id="bar-plot", figure=fig1)
                ]),
                html.Div(className="chart-column", children=[
                    html.Div(className="pie-chat", children=[
                        dcc.Graph(id="survivors-gender-pie", figure=fig)

                    ]),
                ]),
            ]),
            html.Div(className="row2", children=[
                html.Div(className="attribute-dropdown", children=[
                    html.Label("Select Attribute for Charts above", style={'font-family': "Arial Black"}),
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

            html.Div(className="row3", children=[
                # add title
                html.H3("Mushroom Edibility and Poisonous by most common physical attributes",
                        style={'font-family': "Arial Black"}),
                html.H4("Class By Odor (Ring 0), Spore-print-color (Ring 1) & Gill-color (Ring 2)",
                        style={'font-family': "Arial Black"}),
                html.Div(className="circle", children=[
                    dcc.Graph(
                        id='circle-plot',
                        figure=fig3,
                    ),
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
    [Output('bar-plot', 'figure'),
     Output('survivors-gender-pie', 'figure'),
     Output('theil-u-heatmap', 'figure'),
     Output('circle-plot', 'figure'),
     Output('corr_heatmap', 'figure'),
     Output('app-container', 'style')],
    [Input('mode-slider', 'value'),
     Input('theme-slider', 'value'),
     Input('attribute-dropdown', 'value')]
)
def update_graphs(mode_value, theme_value, attribute_value):
    pie = update_pie_chart(df, attribute_value)
    bar = update_bar_plot(df, attribute_value, 0)
    sun = create_sunburst(0)
    big_corr = create_big_heatmap(color_blind=0)
    small_corr = create_heatmap(color_blind=0)

    # Modify the hues used in the graphs for color-blind mode
    if mode_value == 1:
        bar = update_bar_plot(df, attribute_value, 1)
        pie.update_traces(marker=dict(colors=color_blind_list))
        # small_corr.update_traces(marker=dict(colors=color_blind_list))
        sun = create_sunburst(1)
        big_corr = create_big_heatmap(color_blind=1)
        small_corr = create_heatmap(color_blind=1)
    # Reset the hues to default for regular mode
    else:
        pie.update_traces(marker=dict(colors=['rgba(31, 119, 180, 0.5)', 'rgba(255, 127, 14, 0.5)']))

    # Apply light theme with white background
    if theme_value == 0:
        bar.update_layout(template="plotly", paper_bgcolor='#ffffff', plot_bgcolor='#ffffff')
        pie.update_layout(template="plotly", paper_bgcolor='#ffffff', plot_bgcolor='#ffffff')
        small_corr.update_layout(template="plotly", paper_bgcolor='#ffffff', plot_bgcolor='#ffffff')
        sun.update_layout(template="plotly", paper_bgcolor='#ffffff', plot_bgcolor='#ffffff')
        big_corr.update_layout(template="plotly", paper_bgcolor='#ffffff', plot_bgcolor='#ffffff')

        return bar, pie, small_corr, sun, big_corr, {'backgroundColor': '#ffffff', 'color': '#002642'}

    # Apply dark theme with black background
    else:
        bar.update_layout(template="plotly_dark", paper_bgcolor='#002642', plot_bgcolor='#002642')
        pie.update_layout(template="plotly_dark", paper_bgcolor='#002642', plot_bgcolor='#002642')
        small_corr.update_layout(template="plotly_dark", paper_bgcolor='#002642', plot_bgcolor='#002642')
        sun.update_layout(template="plotly_dark", paper_bgcolor='#002642', plot_bgcolor='#002642')
        big_corr.update_layout(template="plotly_dark", paper_bgcolor='#002642', plot_bgcolor='#002642')

        pie.update_traces(textfont_color='#FFFFFF')
        sun.update_traces(textfont_color='#FFFFFF')
        big_corr.update_traces(textfont_color='#FFFFFF')
        small_corr.update_traces(textfont_color='#FFFFFF')

        return bar, pie, small_corr, sun, big_corr, {'backgroundColor': '#002642', 'color': '#E5DADA'}


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


if __name__ == '__main__':
    app.run_server(debug=True)
