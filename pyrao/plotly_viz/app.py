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

# Читаем данные

directory = 'C:/Users/Somov.A/Documents/astro/'
path1 = directory + '070712_05_00.pnt'
path2 = directory + 'eq_1_6b_20120706_20130403.txt'
path3 = directory + 'output/'

data = BSAData()
data.read(path1)
data.calibrate(path2)

# Для начала можно вывести порядка 1000 точек в каждом луче

n_samples = data.data.shape[0]-35000
n_channels = 48

data_trunc = data.data[:n_samples,:n_channels, 0]  # Обрезанные данные
times = pd.Series(data.dt[:n_samples])  # Список datetime наблюдений

ch_names = (np.arange(n_channels) + 1).astype(str)  # Названия лучей
# Сетка расположения лучей на графике
domains = np.linspace(1, 0, n_channels + 1)

traces = [Scatter(x=times,
                    y=data_trunc[:, i],
                    xaxis=f'x{i + 1}',
                    yaxis=f'y{i + 1}',
                    line = dict(
                        color = ('rgb(0, 0, 255)'),
                        width = 0.7
                        )
                    ) for i in range(0, n_channels)]
annotations = [Annotation(x=-0.06,
                          y=data_trunc[:, i].mean(),
                          xref='paper',
                          yref=f'y{i + 1}',
                          text=ch_name,
                          # font=Font(size=9),
                          showarrow=False)
                          for i, ch_name in enumerate(ch_names)]

fig = tools.make_subplots(rows=n_channels, cols=1, specs=[[{}]] * n_channels,
                          shared_xaxes=True, shared_yaxes=False,
                          vertical_spacing=-5, print_grid=False);

for i, trace in enumerate(traces):
    fig.append_trace(trace, i + 1, 1)
    fig['layout'].update({f'yaxis{i + 1}': YAxis(
                                {
                                    'domain': np.flip(domains[i:i + 2]),
                                    'showticklabels': False,
                                    'zeroline': False,
                                    'showgrid': False,
                                    'automargin': False
                                }),
                          'showlegend': False,
                          'margin': dict(t = 50)})

fig['layout'].update(autosize=False, height=1000)
fig['layout']['xaxis'].update(side='top')
fig['layout']['xaxis'].update(mirror='allticks', side='bottom')
# fig['layout'].update(annotations=annotations)

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)  # , external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
        html.Div(
            #'Graph',
            #style=,
            [
                html.Label('График интенсивности радиосигнала для телескопа BSA'),
                dcc.Graph(
                    id='example-graph',
                    figure=fig
                )
            ],
            className="columns",
            style={'display': 'inline-block',
                   'vertical-align': 'top',
                   'width': '60%'}),

        html.Div(
            #'Date picker',
            [
                html.Label('Выберите временной промежуток'),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    # start_date=dt(2012, 07, 07),
                    # end_date_placeholder_text='Select a date!'
                )
            ],
            className="columns",
            style={'display': 'inline-block',
                   'vertical-align': 'top',
                   'width': '35%'})
],
style={'marginTop' : '5%',
       'marginLeft' : '5%'})

if __name__ == '__main__':
    app.run_server(debug=True)
