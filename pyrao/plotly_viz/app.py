from datetime import datetime as dt

from pyrao import BSAData

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from plotly import tools
from plotly.graph_objs import Scatter, Scattergl
from plotly.graph_objs.layout import XAxis, YAxis, Annotation, Font

from flask_caching import Cache
from flask_socketio import SocketIO, emit
import os
import time
import uuid
import json


app = dash.Dash(__name__)
cache = Cache(app.server, config={
    # 'CACHE_TYPE': 'redis',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': './',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 50
})

datetime_format = '%d%m%y_%H_00'
datetime_min = "2012-07-07T04:00"
datetime_max = "2012-07-07T04:59"
datetime_init = "2012-07-07T04:00"

def get_data(session_id, new_datetime, old_datetime=None):
    @cache.memoize()
    def query_data(session_id, new_datetime_str):
        """Здесь нужно реализовать взаимодействие с БД и доступ к файлу по id времени"""
        if old_datetime is not None:
            cache.delete_memoized(session_id, old_datetime)

        path1 = '070712_05_00.pnt' # date[8:10]+date[5:7]+date[2:4]+'_'+time[0:2]+'_00.pnt'
        path2 = 'eq_1_6b_20120706_20130403.txt'

        data = BSAData()
        data.read(path1)
        data.calibrate(path2)

        return data.data, data.dt

    new_datetime = pd.to_datetime(new_datetime)
    new_datetime_str = new_datetime.strftime(datetime_format)
    data, datetimes = query_data(session_id, new_datetime_str)

    start_dt = np.datetime64(new_datetime)
    end_dt = start_dt + np.timedelta64(1, 'h')
    start_ix = np.where(datetimes>=start_dt)[0][0]
    end_ix = np.where(datetimes<=end_dt)[0][-1]
    return data[start_ix:end_ix], datetimes[start_ix:end_ix]

def create_traces(data, datetimes, n_channels, use_gradient):
    traces = [Scattergl(x=datetimes,
                      y=data[:, i],
                      xaxis=f'x{i + 1}',
                      yaxis=f'y{i + 1}',
                      line = dict(
                          color = ('rgb(0, 0, 255)'),
                          width = 0.7
                          )
                      ) for i in range(n_channels)]
    if use_gradient:
        for i, trace in enumerate(traces):
            trace.line['color'] = f'rgb({255*i/n_channels}, 0, {255*(1-i/n_channels)})'
    return traces

def create_annotations(data, n_channels):
    return [Annotation(x=-0.06,
                       y=data[:, i].mean(),
                       xref='paper',
                       yref=f'y{i + 1}',
                       text=f'{i + 1}',
                       # font=Font(size=9),
                       showarrow=False)
                       for i in np.arange(n_channels)]

def setup_figure(data, datetimes, use_gradient, show_yaxis_ticks, height):
    n_channels = data.shape[1]
    datetimes = [dt.utcfromtimestamp((datetime + np.timedelta64(3, 'h')).tolist()/1e9)
                 for datetime in datetimes]
    domains = np.linspace(1, 0, n_channels + 1)

    fig = tools.make_subplots(rows=n_channels, cols=1, # specs=[[{}]] * n_channels,
                              shared_xaxes=True, shared_yaxes=True,
                              vertical_spacing=-5, print_grid=False)

    traces = create_traces(data, datetimes, n_channels, use_gradient)
    for i, trace in enumerate(traces):
        fig.append_trace(trace, i + 1, 1)
        fig['layout'].update({f'yaxis{i + 1}': YAxis(
                                    {
                                        'domain': np.flip(domains[i:i + 2], axis=0),
                                        'showticklabels': show_yaxis_ticks,
                                        'zeroline': False,
                                        'showgrid': False,
                                        'automargin': False
                                    }),
                            'showlegend': False,
                            'margin': dict(t = 50)
                            })
    if not show_yaxis_ticks:
        annotations = create_annotations(data, n_channels)
        fig['layout'].update(annotations=annotations)

    fig['layout'].update(autosize=False, height=height)
    # fig['layout']['xaxis'].update(side='top')
    # fig['layout']['xaxis'].update(tickformat='%H:%M:%S:%L')
    # fig['layout']['xaxis'].update(mirror='allticks', side='bottom')

    return fig

def serve_layout():
    data, datetimes = get_data(session_id, datetime_init)
    return html.Div(id='main-div', children=[
        dcc.Store(id='session-id', data=str(uuid.uuid4()), storage_type='session'),
        dcc.Store(id='current-ray', data=0, storage_type='session'),
        html.Div(pd.to_datetime(datetime_init).strftime(datetime_format),
                 id='current-dt', style={'display': 'none'}),
        html.Div(
            [
                html.H3('Фильтры:', style={'marginLeft' : '50px'}),
                html.Div(
                    dcc.Dropdown(
                        id="filters-dropdown",
                        options=[
                            {'label': 'Multiply', 'value': 'Multiply'},
                            {'label': 'Cut', 'value': 'Cut'},
                        ],
                        value=[],
                        multi=True
                    ), style={'marginLeft' : '50px', 'width': "500px"}
                ),
                html.Br(),
                html.Label('График интенсивности радиосигнала для телескопа BSA',
                           style={'marginLeft' : '50px'}),
                dcc.Graph(
                    id='main-graph',
                    figure=setup_figure(data[:, :, 0],
                                        datetimes,
                                        use_gradient=False,
                                        show_yaxis_ticks=False,
                                        height=1000)
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
                    dcc.Input(id='datetime-picker',
                              type='datetime-local',
                              style={
                                  'font-weight': '200',
                                  'font-size': '18px',
                                  'line-height': '28px',
                                  'margin': '0',
                                  'padding': '8px',
                                  'background': '#fff',
                                  'position': 'relative',
                                  'display': 'inline-block',
                                  'width': '300px',
                                  'vertical-align': 'middle'
                              },
                              min=datetime_min,
                              max=datetime_max,
                              value=datetime_init
                    ),
                    html.Br(),
                    html.Br(),
                    html.Div(id='datetime-label')
                ], style={'marginLeft' : '50px'}),
                html.Br(),
                dcc.Graph(
                    id='one-ray-graph',
                    figure=setup_figure(data[:, 0, 0].reshape(-1, 1),
                                        datetimes,
                                        use_gradient=False,
                                        show_yaxis_ticks=True,
                                        height=300)
                ),
                dcc.Graph(
                    id='freq-graph',
                    figure=setup_figure(data[:, 0, 1:],
                                        datetimes,
                                        use_gradient=True,
                                        show_yaxis_ticks=False,
                                        height=500)
                ),
                html.Div(id='placeholder')
            ],
        className="columns",
        style={'display': 'inline-block',
            'vertical-align': 'top',
            'width': '50%'})
    ])

app.layout = serve_layout
"""
@app.callback(Output('main-div', 'children'),
              [Input('session-id', 'children'),
               Input('current-dt', 'children'),
               Input('main-div', 'children'),
               Input('current-ray', 'children')])
def update_datetime(session_id, current_dt, main_div, new_ray):
    data, datetimes = get_data(session_id, current_dt)
    main_div[-2]['children'][-1]['figure'] = setup_figure(data[:, :, 0],
                                                          datetimes,
                                                          use_gradient=False,
                                                          show_yaxis_ticks=False,
                                                          height=1000)
    main_div[-1]['children'][-3]['figure'] = setup_figure(data[:, new_ray, 0],
                                                          datetimes,
                                                          use_gradient=False,
                                                          show_yaxis_ticks=True,
                                                          height=300)
    main_div[-1]['children'][-2]['figure'] = setup_figure(data[:, new_ray, 1:],
                                                          datetimes,
                                                          use_gradient=True,
                                                          show_yaxis_ticks=False,
                                                          height=500)
    print(main_div)
    return main_div
"""
@app.callback(Output('current-dt', 'children'),
              [Input('session-id', 'children'),
               Input('datetime-picker', 'value')])
def update_datetime(session_id, value, current_dt):
    get_data(session_id, value, current_dt)
    print(current_dt)
    return None#current_dt#pd.to_datetime(value).strftime(datetime_format)

"""@app.callback(Output('main-graph', 'figure'),
              [Input('session-id', 'children'),
               Input('datetime-picker', 'value')])
def update_main_graph(session_id, value):
    data, datetimes = get_data(session_id, value)
    data = data[:, :, 0]
    return setup_figure(data,
                        datetimes,
                        use_gradient=False,
                        show_yaxis_ticks=False,
                        height=1000)

@app.callback(Output('one-ray-graph', 'figure'),
              [Input('session-id', 'children'),
               Input('datetime-picker', 'value'),
               Input('current-ray', 'children')])
def update_one_ray_graph(session_id, value, new_ray):
    data, datetimes = get_data(session_id, value)
    data = data[:, new_ray, 0].reshape(-1, 1)
    fig = setup_figure(data,
                       datetimes,
                       use_gradient=False,
                       show_yaxis_ticks=True,
                       height=300)
    return fig

@app.callback(Output('freq-graph', 'figure'),
              [Input('session-id', 'children'),
               Input('datetime-picker', 'value'),
               Input('current-ray', 'children')])
def update_freq_graph(session_id, value, new_ray):
    data, datetimes = get_data(session_id, value)
    data = data[:, new_ray, 1:]
    fig = setup_figure(data,
                       datetimes,
                       use_gradient=True,
                       show_yaxis_ticks=False,
                       height=500)
    return fig"""
#
# @app.callback(Output('current-ray', 'children'),
#               [Input('main-graph', 'clickData')])
# def update_ray(clickData):
#     return clickData['points'][0]['curveNumber'] if clickData is not None else 0

# Callbacks for synchronous updating xaxis on all graphs
# @app.callback(Output('main-graph', 'figure'),
#               [Input('main_graph', 'relayoutData')])
# def

@app.callback(
    dash.dependencies.Output('placeholder', 'children'),
    [dash.dependencies.Input('filters-dropdown', 'value')])
def update_output(value):
    print(value)


if __name__ == '__main__':
    app.run_server(debug=True)
