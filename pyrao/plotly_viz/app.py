from datetime import datetime as dt

from pyrao import BSAData

import pandas as pd
import numpy as np

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from plotly import tools
from plotly.graph_objs import Scatter, Scattergl, Figure
from plotly.graph_objs.layout import XAxis, YAxis, Annotation, Font

from flask_caching import Cache
import os
import time
import uuid
import json

"""
СДЕЛАТЬ ВСЕ ГРАФИКИ SUBPLOT'ами
УДАЛЯТЬ КАЛИБРАЦИОННЫЕ ПЕРИОДЫ
"""

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

def get_data(session_id, date):
    old_date = date[0]
    new_date = date[1]
    @cache.memoize()
    def query_data(session_id, new_date):
        """Здесь нужно реализовать взаимодействие с БД и доступ к файлу по id времени"""
        if old_date is not None and old_date != new_date:
            print('Deleting cache:', session_id, old_date)
            cache.delete_memoized(query_data, session_id, old_date)

        path1 = '070712_05_00.pnt' # date[8:10]+date[5:7]+date[2:4]+'_'+time[0:2]+'_00.pnt'
        path2 = 'eq_1_6b_20120706_20130403.txt'

        data = BSAData()
        data.read(path1)
        data.calibrate(path2)

        return data.data, data.dt

    data, datetimes = query_data(session_id, new_date)

    new_datetime = pd.to_datetime(new_date)
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

def get_figures(data, datetimes, ray):
    fig1 = setup_figure(data[:, :, 0],
                      datetimes,
                      use_gradient=False,
                      show_yaxis_ticks=False,
                      height=1000)
    fig2 = setup_figure(data[:, ray, 0].reshape(-1, 1),
                      datetimes,
                      use_gradient=False,
                      show_yaxis_ticks=True,
                      height=300)
    fig3 = setup_figure(data[:, ray, 1:],
                      datetimes,
                      use_gradient=True,
                      show_yaxis_ticks=False,
                      height=500)
    return fig1, fig2, fig3

def serve_layout():
    session_id = str(uuid.uuid4())
    data, datetimes = get_data(session_id, [datetime_init, datetime_init])
    fig1, fig2, fig3 = get_figures(data, datetimes, 0)
    return html.Div(id='main-div', children=[
        dcc.Store(id='session-id', storage_type='session'),
        dcc.Store(id='current-ray', storage_type='session'),
        dcc.Store(id='datetime', storage_type='session'),
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
                    figure=fig1
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
                    figure=fig2
                ),
                dcc.Graph(
                    id='freq-graph',
                    figure=fig3
                ),
                html.Div(id='placeholder')
            ],
        className="columns",
        style={'display': 'inline-block',
            'vertical-align': 'top',
            'width': '50%'})
    ])

app.layout = serve_layout

@app.callback([Output('main-graph', 'figure'),
               Output('one-ray-graph', 'figure'),
               Output('freq-graph', 'figure')],
              [Input('datetime', 'data'),
               Input('current-ray', 'data'),
               Input('main-graph', 'relayoutData')],
              [State('session-id', 'data'),
               State('datetime', 'data'),
               State('current-ray', 'data'),
               State('main-graph', 'figure'),
              State('one-ray-graph', 'figure'),
              State('freq-graph', 'figure')])
def update_graphs(date, ray, relayoutData, session_id, old_date, old_ray, f1, f2, f3):
    date = date if date is not None else old_date
    date = date if date is not None else [datetime_init, datetime_init]
    ray = ray if ray is not None else old_ray
    ray = ray if ray is not None else 0

    print('upd graphs', date, session_id, ray)
    print(relayoutData)
    if f1 is not None and f2 is not None and f3 is not None and relayoutData is not None:
        f1['layout']['xaxis']['range'] = [relayoutData['xaxis.range[0]']]
        f1['layout']['xaxis']['range'].append(relayoutData['xaxis.range[1]'])
        f2['layout']['xaxis']['range'] = [relayoutData['xaxis.range[0]']]
        f2['layout']['xaxis']['range'].append(relayoutData['xaxis.range[1]'])
        f3['layout']['xaxis']['range'] = [relayoutData['xaxis.range[0]']]
        f3['layout']['xaxis']['range'].append(relayoutData['xaxis.range[1]'])
        return f1, f2, f3

    data, datetimes = get_data(session_id, date)

    return get_figures(data, datetimes, ray)

@app.callback(Output('session-id', 'data'),
              [Input('session-id', 'modified_timestamp')],
              [State('session-id', 'data')])
def update_session_id(modified_timestamp, data):
    print('sess id', modified_timestamp, data)
    if modified_timestamp is None:
        return data if data is not None else str(uuid.uuid4())
    return data if data is not None else str(uuid.uuid4())

@app.callback(Output('current-ray', 'data'),
              [Input('main-graph', 'clickData')],
              [State('session-id', 'data')])
def update_ray(clickData, session_id):
    print('ray', clickData['points'][0]['curveNumber'] if clickData is not None else None, session_id)
    return clickData['points'][0]['curveNumber'] if clickData is not None else 0

@app.callback(Output('datetime', 'data'),
              [Input('datetime-picker', 'value')],
              [State('session-id', 'data'),
               State('datetime', 'data')])
def update_datetime(value, session_id, data):
    print('dt', value, data, session_id)
    return (data[1], value) if data is not None else (value, value)

@app.callback(
    Output('placeholder', 'children'),
    [Input('filters-dropdown', 'value')])
def update_output(value):
    print(value)

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

# Callbacks for synchronous updating xaxis on all graphs

# def


if __name__ == '__main__':
    app.run_server(debug=True)
