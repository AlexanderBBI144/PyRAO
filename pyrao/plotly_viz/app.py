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
app.config['suppress_callback_exceptions']=True
# Читаем данные
n_samples = 1000

def get_data(date_start, date_end):
    path1 = '070712_05_00.pnt'
    path2 = 'eq_1_6b_20120706_20130403.txt'
    path3 = './output/'

    data = BSAData()
    data.read(path1)
    data.calibrate(path2)

    return data

def build_main_graph(data):
    n_channels = 48
    # Для начала можно вывести порядка 1000 точек в каждом луче
    data_trunc = data.data[:n_samples,:n_channels, 0]  # Обрезанные данные
    # data_trunc[500:510, 5] = 2000
    times = pd.Series(data.dt[:n_samples])  # Список datetime наблюдений

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

def build_detailed_rays_graph(data, ray_number=1):
    n_channels = 7
    # Для начала можно вывести порядка 1000 точек в каждом луче
    data_trunc = data.data[:n_samples, ray_number, :]  # Обрезанные данные
    times = pd.Series(data.dt[:n_samples])  # Список datetime наблюдений

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

def build_one_ray_graph(data, ray_number=1):
    data_trunc = data.data[:n_samples, ray_number, 0]  # Обрезанные данные
    times = pd.Series(data.dt[:n_samples])  # Список datetime наблюдений

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


# Создание приложения
def build_app(main_fig, second_fig, third_fig):
    app.layout = html.Div(children=[
            html.Div(
                #'Graph',
                #style=,
                [
                    html.Label('График интенсивности радиосигнала для телескопа BSA', style={'marginLeft' : '50px'}),
                    dcc.Graph(
                        id='beams-graph',
                        figure=main_fig
                    )
                ],
                className="columns",
                style={'display': 'inline-block',
                    'vertical-align': 'top',
                    'width': '50%'}),

            html.Div(
                [
                    html.Label('Выберите дату и время: ', style={'marginLeft' : '50px'}),
                    html.Br(),
                    html.Br(),
                    html.Div([
                        dcc.DatePickerSingle(
                            id='my-date-picker-single',
                            min_date_allowed=dt(2012, 7, 6),
                            initial_visible_month=dt(2012, 7, 7),
                            date=dt(2012, 7, 7)
                        ),
                        dcc.Input(id='my-time-picker-single',
                                    type='time',
                                    style={
                                        'font-weight': '200',
                                        'font-size': '18px',
                                        'line-height': '24px',
                                        'color': '#757575',
                                        'margin': '0',
                                        'padding': '8px',
                                        'background': '#fff',
                                        'position': 'relative',
                                        'display': 'inline-block',
                                        'width': '130px',
                                        'vertical-align': 'middle'
                                    },
                                    value=dt.strptime('10:30', '%H:%M')
                        ),
                        html.Br(),
                        html.Br(),
                        html.Div(id='output-container-date-picker-single')
                    ], style={'marginLeft' : '50px'}),
                    html.Br(),
                    html.Br(),
                    html.Label('луч 1', id='one-ray-label' ,style={'marginLeft' : '50px'}),
                    dcc.Graph(
                        id='beams-graph3',
                        figure=third_fig
                    ),
                    html.Br(),
                    html.Label('луч 1 в 6 частотах', style={'marginLeft' : '50px'}),
                    html.Br(),
                    dcc.Graph(
                        id='beams-graph2',
                        figure=second_fig
                    )
                ],
                className="columns",
                style={'display': 'inline-block',
                    'vertical-align': 'top',
                    'width': '50%'})
    ],
    style={'marginTop' : '5%'})
    

@app.callback(
    dash.dependencies.Output('output-container-date-picker-single', 'children'),
    [dash.dependencies.Input('my-time-picker-single', 'value')
    ])
def update_output(value):
    print(value)
    # string_prefix = 'Данные показаны для: '
    # if date is not None:
    #     date = dt.strptime(date, '%Y-%m-%d')
    #     date_string = date.strftime('%B %d, %Y')
    #     return string_prefix + date_string
# @app.callback(
#     dash.dependencies.Output('beams-graph', 'figure'),
#     [dash.dependencies.Input('apply-filter-button', 'n_clicks')])
# def apply_math_filter(n_clicks):
#     print("filter applied")   
#     coef = 2 if n_clicks % 2 == 0 else 1/2
#     for i, trace in enumerate(fig['data']):
#         fig['data'][i]['y'] = trace['y'] * coef
#         fig['layout']['annotations'][i]['y'] = fig['layout']['annotations'][i]['y'] * coef
#     return fig


if __name__ == '__main__':
    data = get_data(0,0)
    main_fig = build_main_graph(data)
    second_fig = build_detailed_rays_graph(data,1)
    third_fig = build_one_ray_graph(data,1)
    build_app(main_fig, second_fig, third_fig)
    app.run_server(debug=True)
