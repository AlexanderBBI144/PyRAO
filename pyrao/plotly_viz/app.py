from datetime import datetime as dt

from pyrao import BSAData

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly import tools
from plotly.graph_objs import Scatter, Scattergl
from plotly.graph_objs.layout import XAxis, YAxis, Annotation, Font

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)  # , external_stylesheets=external_stylesheets)

# Читаем данные
n_samples = 1000

date_str = "2012-07-07"
def get_date():
    return dt.strptime(date_str, '%Y-%m-%d')

time = "05:00"
data = None
def get_data():
    return data
def update_data(date, time):
    global data
    path1 = '070712_05_00.pnt' #date[8:10]+date[5:7]+date[2:4]+'_'+time[0:2]+'_00.pnt'
    path2 = 'eq_1_6b_20120706_20130403.txt'
    path3 = './output/'

    _data = BSAData()
    _data.read(path1)
    _data.calibrate(path2)
    data = _data

# ---------- Layout

app.layout = html.Div(children=[
    html.Div(
        [
            html.H3('Фильтры:', style={'marginLeft' : '50px'}),
            html.Div(dcc.Dropdown(
                id="filters-dropdown",
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montreal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                value=['MTL', 'NYC'],
                multi=True
            ), style={'marginLeft' : '50px', 
                        'width': "500px"}),
            html.Br(),
            html.Label('График интенсивности радиосигнала для телескопа BSA', style={'marginLeft' : '50px'}),
            dcc.Graph(
                id='graph-1'
            )
        ],
        className="columns",
        style={'display': 'inline-block',
            'vertical-align': 'top',
            'width': '50%'}),

    html.Div(
        [
            html.H3('Выберите дату и время: ', style={'marginLeft' : '50px'}),
            html.Div([
                dcc.DatePickerSingle(
                    id='date-picker',
                    min_date_allowed=get_date(),
                    initial_visible_month=get_date(),
                    date=get_date()
                ),
                dcc.Input(id='time-picker',
                            type='time',
                            style={
                                'font-weight': '200',
                                'font-size': '18px',
                                'line-height': '28px',
                                'margin': '0',
                                'padding': '8px',
                                'background': '#fff',
                                'position': 'relative',
                                'display': 'inline-block',
                                'width': '100px',
                                'vertical-align': 'middle'
                            },
                            value=time
                ),
                html.Button('Показать', id='datetime-button', style={
                                'width': '120px',
                                'height': '50px',
                            }),
                html.Br(),
                html.Br(),
                html.Div(id='datetime-label')
            ], style={'marginLeft' : '50px'}),
            html.Br(),
            html.H3('Задайте номер луча: ', style={'marginLeft' : '50px'}),
            html.Div([
                dcc.Input(id='ray-number-input',
                            type='text',
                            style={
                                'font-weight': '200',
                                'font-size': '18px',
                                'line-height': '32px',
                                'margin': '0',
                                'padding': '8px',
                                'background': '#fff',
                                'position': 'relative',
                                'display': 'inline-block',
                                'width': '100px',
                                'vertical-align': 'middle'
                            },
                            value=1
                ),
                html.Button('Показать', id='ray-button', style={
                                'width': '120px',
                                'height': '50px',
                            })
            ], style={'marginLeft' : '50px'}),
            dcc.Graph(
                id='graph-2'
            ),
            dcc.Graph(
                id='graph-3'
            ),
            html.Div(id='placeholder')
        ],
    className="columns",
    style={'display': 'inline-block',
        'vertical-align': 'top',
        'width': '50%'})
])
    
# ---------- Callbacks

# 1. updating datetime label

@app.callback(
    dash.dependencies.Output('datetime-label', 'children'),
    [dash.dependencies.Input('datetime-button', 'n_clicks')],
    [dash.dependencies.State('date-picker', 'date'), dash.dependencies.State('time-picker', 'value')])
def update_datetime_label(n_clicks, date, value):
    if date is not None and value is not None :
        string_prefix = 'Данные показаны для: '
        string = string_prefix + date + " " + value
        date = date
        time = value
        print(string)
        return string

# 2. reading data from date time and updating 1st graph

@app.callback(
    dash.dependencies.Output('graph-1', 'figure'),
    [dash.dependencies.Input('datetime-button', 'n_clicks')],
    [dash.dependencies.State('date-picker', 'date'), dash.dependencies.State('time-picker', 'value')])
def update_graph1(n_clicks, date, value):
    if date is not None and value is not None :
        # string_prefix = 'Данные показаны для: '
        # string = string_prefix + date + " " + value
        _date = get_date()
        update_data(_date, time)
        _data = get_data()
        fig = setup_main_praph(_data)
        return fig


# 3. updating 2nd graph

@app.callback(
    dash.dependencies.Output('graph-2', 'figure'),
    [dash.dependencies.Input('ray-button', 'n_clicks')],
    [dash.dependencies.State('ray-number-input', 'value')])
def update_graph2(n_clicks, value):
    if value is not None:
        print(f"Ray number changed to: {value}")
        _data = get_data()
        fig = setup_one_ray_graph(_data, int(value))
        return fig

# 4. updating 3nd graph

@app.callback(
    dash.dependencies.Output('graph-3', 'figure'),
    [dash.dependencies.Input('ray-button', 'n_clicks')],
    [dash.dependencies.State('ray-number-input', 'value')])
def update_graph3(n_clicks, value):
    if value is not None:
        print(f"Ray number changed to: {value}")
        _data = get_data()
        fig = setup_detailed_ray_graph(_data, int(value))
        return fig


@app.callback(
    dash.dependencies.Output('placeholder', 'children'),
    [dash.dependencies.Input('filters-dropdown', 'value')])
def update_output(value):
    print(value)



# ---------- Functions

def setup_main_praph(_data):
    n_channels = 48
    # Для начала можно вывести порядка 1000 точек в каждом луче
    data_trunc = _data.data[:n_samples,:n_channels, 0]  # Обрезанные данные
    # data_trunc[500:510, 5] = 2000
    times = pd.Series(_data.dt[:n_samples])  # Список datetime наблюдений

    ch_names = (np.arange(n_channels) + 1).astype(str)  # Названия лучей
    # Сетка расположения лучей на графике
    domains = np.linspace(1, 0, n_channels + 1)

    # Список графиков лучей
    traces = [Scattergl(x=times,
                        y=data_trunc[:, i],
                        xaxis=f'x{i + 1}',
                        yaxis=f'y{i + 1}',
                        line = dict(
                            color = ('rgb(0, 0, 255)'),
                            width = 0.7
                            )
                        ) for i in range(0, n_channels)]

    # Список номеров лучей
    annotations = [Annotation(x=-0.06,
                            y=data_trunc[:, i].mean(),
                            xref='paper',
                            yref=f'y{i + 1}',
                            text=ch_name,
                            # font=Font(size=9),
                            showarrow=False)
                            for i, ch_name in enumerate(ch_names)]

    # Создание графика

    fig = tools.make_subplots(rows=n_channels, cols=1, specs=[[{}]] * n_channels,
                            shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=-5, print_grid=False)

    for i, trace in enumerate(traces):
        fig.append_trace(trace, i + 1, 1)
        fig['layout'].update({f'yaxis{i + 1}': YAxis(
                                    {
                                        'domain': np.flip(domains[i:i + 2], axis=0),
                                        'showticklabels': False,
                                        'zeroline': False,
                                        'showgrid': False,
                                        'automargin': False
                                    }),
                            'showlegend': False,
                            'margin': dict(t = 50)
                            })

    fig['layout'].update(autosize=False, height=1000)
    fig['layout']['xaxis'].update(side='top')
    fig['layout']['xaxis'].update(mirror='allticks', side='bottom')
    fig['layout'].update(annotations=annotations)
    return fig


def setup_one_ray_graph(_data, ray_number=1):
    ray_number = ray_number - 1
    data_trunc = _data.data[:n_samples, ray_number, 0]  # Обрезанные данные
    times = pd.Series(_data.dt[:n_samples])  # Список datetime наблюдений

    # Сетка расположения лучей на графике
    domain = [1, 0]

    # Список графиков лучей
    trace = Scattergl(x=times,
                        y=data_trunc,
                        xaxis=f'x1',
                        yaxis=f'y1',
                        line = dict(
                            color = ('rgb(0, 0, 255)'),
                            width = 0.7
                            )
                        )

    # Создание графика

    fig = tools.make_subplots(rows=1, cols=1, specs=[[{}]],
                            shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=-5, print_grid=False)

    
    fig.append_trace(trace, 1, 1)
    fig['layout'].update({f'yaxis1': YAxis(
                                {
                                    'domain': np.flip(domain, axis=0),
                                    'showticklabels': True,
                                    'zeroline': False,
                                    'showgrid': False,
                                    'automargin': False
                                }),
                        'showlegend': False,
                        'margin': dict(t = 50)
                        })

    fig['layout'].update(autosize=False, height=300)
    fig['layout']['xaxis'].update(side='top')
    fig['layout']['xaxis'].update(mirror='allticks', side='bottom')
    return fig    


def setup_detailed_ray_graph(_data, ray_number=1):
    ray_number = ray_number - 1
    n_channels = 7
    # Для начала можно вывести порядка 1000 точек в каждом луче
    data_trunc = _data.data[:n_samples, ray_number, :]  # Обрезанные данные
    times = pd.Series(_data.dt[:n_samples])  # Список datetime наблюдений

    ch_names = (np.arange(n_channels) + 1).astype(str)  # Названия лучей
    # Сетка расположения лучей на графике
    domains = np.linspace(1, 0, n_channels + 1)

    # Список графиков лучей
    traces = [Scattergl(x=times,
                        y=data_trunc[:, i],
                        xaxis=f'x{i + 1}',
                        yaxis=f'y{i + 1}',
                        line = dict(
                            color = (f'rgb({255*i/n_channels}, 0, {255*(1-i/n_channels)})'),
                            width = 0.7
                            )
                        ) for i in range(0, n_channels)]

    # Список номеров лучей
    annotations = [Annotation(x=-0.06,
                            y=data_trunc[:, i].mean(),
                            xref='paper',
                            yref=f'y{i + 1}',
                            text=ch_name,
                            # font=Font(size=9),
                            showarrow=False)
                            for i, ch_name in enumerate(ch_names)]

    # Создание графика

    fig = tools.make_subplots(rows=n_channels, cols=1, specs=[[{}]] * n_channels,
                            shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=-5, print_grid=False)

    for i, trace in enumerate(traces):
        fig.append_trace(trace, i + 1, 1)
        fig['layout'].update({f'yaxis{i + 1}': YAxis(
                                    {
                                        'domain': np.flip(domains[i:i + 2], axis=0),
                                        'showticklabels': False,
                                        'zeroline': False,
                                        'showgrid': False,
                                        'automargin': False
                                    }),
                            'showlegend': False,
                            'margin': dict(t = 50)
                            })

    fig['layout'].update(autosize=False, height=500)
    fig['layout']['xaxis'].update(side='top')
    fig['layout']['xaxis'].update(mirror='allticks', side='bottom')
    fig['layout'].update(annotations=annotations)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
