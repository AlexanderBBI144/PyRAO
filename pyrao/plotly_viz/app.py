from datetime import datetime as dt

import pandas as pd
import numpy as np

import dash
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
import dash_core_components as dcc
import dash_html_components as html
import uuid
import logging

from flask_caching import Cache
# from pyrao.plotly_viz.data import get_data, get_available_dates
# from pyrao.plotly_viz.figure import get_figure1, get_figure2, get_figure3
# from pyrao.plotly_viz.figure import get_figures
from data import get_data, get_available_dates
from figure import get_figure1, get_figure2, get_figure3
from figure import get_figures

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG)

logger = logging.getLogger(name="pyrao.plotly_viz.app")

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

DATES = get_available_dates()
DEFAULT_DATE = DATES[-1].date().strftime("%Y-%m-%d")
DEFAULT_HOUR = DATES[-1].hour
DEFAULT_MINUTE = 0
DEFAULT_RAY = 0
# data, datetimes = get_data(None, DEFAULT_DATE, DEFAULT_HOUR, DEFAULT_MINUTE, stand=1)
# DEFAULT_FIGURES = get_figures(data, datetimes, DEFAULT_RAY)


def serve_layout():
    session_id = str(uuid.uuid4())


    logger.info("Forming layout")
    layout = html.Div(id='body', children=[
        html.Header(children=[
            dcc.Store(
                id='session-id',
                storage_type='session',
                data=session_id
            ),
            dcc.Store(
                id='curr-ray',
                storage_type='session',
                data=DEFAULT_RAY
            )
        ]),
        html.Main(children=[
            html.Div(
                id="datetime-div",
                children=[
                    html.P("Выберите дату и время:"),
                    dcc.DatePickerSingle(
                        id="date",
                        min_date_allowed=DATES[0],
                        max_date_allowed=DATES[-1],
                        date=DEFAULT_DATE
                    ),
                    daq.NumericInput(
                        #"ч.",
                        id="hour",
                        min=0,
                        max=23,
                        value=DEFAULT_HOUR
                    ),
                    daq.NumericInput(
                        #"м.",
                        id="minute",
                        min=0,
                        max=59,
                        value=DEFAULT_MINUTE
                    ),
                    html.Button("Обновить", id="refresh")
                ]
            ),
            html.Div(
                id="filter-div",
                children=[
                    html.P("Укажите фильтры:"),
                    dcc.Dropdown(
                        id="filters-dropdown",
                        options=[
                            {'label': 'Multiply', 'value': 'Multiply'},
                            {'label': 'Cut', 'value': 'Cut'},
                        ],
                        value=[],
                        multi=True
                    ),
                    html.Button("Применить", id="apply")
                ]
            ),
            html.Div(
                id="main-fig-col",
                children=[
                    html.P("Многолучевая диаграмма"),
                    dcc.Loading(
                        id="loading-main",
                        children=[dcc.Graph(id="main-fig")]
                    )
                ]
            ),
            html.Div(
                id="one-ray-fig-col",
                children=[
                    html.P("Однолучевая диаграмма"),
                    dcc.Loading(
                        id="loading-one-ray",
                        children=[dcc.Graph(id="one-ray-fig")]
                    )
                ]
            ),
            html.Div(
                id="freq-fig-col",
                children=[
                    html.P("Частотная диаграмма"),
                    dcc.Loading(
                        id="loading-freq",
                        children=[dcc.Graph(id="freq-fig")]
                    )
                ]
            ),
        ]),
        html.Footer(
            children=[
                html.Label("Copyright 2019"),
                dcc.Link("Alexander Somov", href="https://github.com/AlexanderBBI144/PyRAO")
            ]
        )
    ])

    logger.info("Layout formed")

    return layout


app.layout = serve_layout


# @app.callback([Output('main-graph', 'figure'),
#                Output('one-ray-graph', 'figure'),
#                Output('freq-graph', 'figure')],
#               [Input('datetime', 'data'),
#                Input('current-ray', 'data'),
#                Input('main-graph', 'relayoutData')],
#               [State('session-id', 'data'),
#                State('datetime', 'data'),
#                State('current-ray', 'data'),
#                State('main-graph', 'figure'),
#               State('one-ray-graph', 'figure'),
#               State('freq-graph', 'figure')])
# def update_graphs(date, ray, relayoutData, session_id, old_date, old_ray,
#                   f1, f2, f3):
#     date = date if date is not None else old_date
#     date = date if date is not None else DEFAULT_DATE
#     ray = ray if ray is not None else old_ray
#     ray = ray if ray is not None else 0
#
#     logger.info('upd graphs', date, session_id, ray)
#     logger.info(relayoutData)
#     if f1 is not None and f2 is not None and f3 is not None and \
#             relayoutData is not None:
#         f1['layout']['xaxis']['range'] = [relayoutData['xaxis.range[0]']]
#         f1['layout']['xaxis']['range'].append(relayoutData['xaxis.range[1]'])
#         f2['layout']['xaxis']['range'] = [relayoutData['xaxis.range[0]']]
#         f2['layout']['xaxis']['range'].append(relayoutData['xaxis.range[1]'])
#         f3['layout']['xaxis']['range'] = [relayoutData['xaxis.range[0]']]
#         f3['layout']['xaxis']['range'].append(relayoutData['xaxis.range[1]'])
#         return f1, f2, f3
#
#     data, datetimes = get_data(session_id, date)
#
#     return get_figures(data, datetimes, ray)

# Не нужно, т.к. значение по умолчанию может быть корректно получено
# (нигде не встречается Output(session_id, 'data'))
# https://dash.plot.ly/dash-core-components/store
#
# @app.callback(Output('session-id', 'data'),
#               [Input('session-id', 'modified_timestamp')],
#               [State('session-id', 'data')])
# def update_session_id(modified_timestamp, data):
#     logger.info('sess id', modified_timestamp, data)
#     if modified_timestamp is None:
#         return data if data is not None else str(uuid.uuid4())
#     return data if data is not None else str(uuid.uuid4())


@app.callback(Output('curr-ray', 'data'),
              [Input('main-fig', 'clickData')],
              [State('session-id', 'data')])
def update_ray(clickData, session_id):
    if clickData is not None:
        ray_n = clickData['points'][0]['curveNumber']
        logger.info(f"ray {ray_n} {session_id}")
        return clickData['points'][0]['curveNumber']
    logger.info("ray is None")
    return 0

#
# @app.callback(Output('datetime', 'data'),
#               [Input('datetime-picker', 'value')],
#               [State('session-id', 'data'),
#                State('datetime', 'data')])
# def update_datetime(value, session_id, data):
#     logger.info('dt', value, data, session_id)
#     return (data[1], value) if data is not None else (value, value)

#
# @app.callback(
#     Output('placeholder', 'children'),
#     [Input('filters-dropdown', 'value')])
# def update_output(value):
#     logger.info(f"filters-dropdown {value}: {type(value)}")


# @app.callback(Output('main-fig', 'figure'),
#               [Input('date', 'date')],
#               [State('session-id', 'data')])
# def update_main_graph(date, session_id):
#     logger.info(f"datetime-picker {date}: {type(date)}")
#     data, datetimes = get_data(session_id, date)
#     return get_figure1(data, datetimes)

@app.callback([Output('main-fig', 'figure'),
               Output('one-ray-fig', 'figure'),
               Output('freq-fig', 'figure')],
              [Input('refresh', 'n_clicks'),
               Input('curr-ray', 'data')],
              [State('session-id', 'data'),
               State('date', 'date'),
               State('hour', 'value'),
               State('minute', 'value')])
def update_graphs(n, new_ray, session_id, date, hour, minute):
    pr.dump_stats('log.txt')
    data, datetimes = get_data(session_id, date, hour, minute, stand=1)
    fig1 = get_figure1(data, datetimes)
    fig2 = get_figure2(data, datetimes, new_ray)
    fig3 = get_figure3(data, datetimes, new_ray)
    logger.info("All figures set up. Refreshing layout")
    pr.enable()
    return fig1, fig2, fig3
# #
# @app.callback(Output('freq-graph', 'figure'),
#               [Input('session-id', 'data'),
#                Input('datetime-picker', 'value'),
#                Input('current-ray', 'data')])
# def update_freq_graph(session_id, value, new_ray):
#     data, datetimes = get_data(session_id, value)
#     return get_figure3(data, datetimes, new_ray)
#

if __name__ == '__main__':
    app.run_server(debug=True)
