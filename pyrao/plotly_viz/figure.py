
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
from plotly.graph_objs import Scatter, Scattergl, Figure
from plotly.graph_objs.layout import XAxis, YAxis, Annotation, Font
from plotly import tools


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG)

logger = logging.getLogger(name="pyrao.plotly_viz.figure")


def setup_figure(data, datetimes, use_gradient, show_yaxis_ticks, height):
    """Create figure."""
    n_channels = data.shape[1]
    datetimes = [
        dt.utcfromtimestamp((datetime + np.timedelta64(3, 'h')).tolist()/1e9)
        for datetime in datetimes
    ]
    domains = np.linspace(1, 0, n_channels + 1)

    fig = tools.make_subplots(
        rows=n_channels,
        cols=1,
        # specs=[[{}]] * n_channels,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=-5,
        print_grid=False
    )

    traces = create_traces(data, datetimes, n_channels, use_gradient)
    for i, trace in tqdm(enumerate(traces), desc='Appending traces to figure'):
        fig.append_trace(trace, i + 1, 1)
        fig['layout'].update({
            f'yaxis{i + 1}': YAxis({
                'domain': np.flip(domains[i:i + 2], axis=0),
                'showticklabels': show_yaxis_ticks,
                'zeroline': False,
                'showgrid': False,
                'automargin': False
            }),
            'showlegend': False,
            'margin': {'t': 50}
        })

    if not show_yaxis_ticks:
        annotations = create_annotations(data, n_channels)
        fig['layout'].update(annotations=annotations)

    fig['layout'].update(autosize=False, height=height)
    # fig['layout']['xaxis'].update(side='top')
    # fig['layout']['xaxis'].update(tickformat='%H:%M:%S:%L')
    # fig['layout']['xaxis'].update(mirror='allticks', side='bottom')

    logger.info("Figure set up")
    return fig


def create_traces(data, datetimes, n_channels, use_gradient):
    logger.info("Creating traces")
    traces = [
        Scattergl(
            x=datetimes,
            y=data[:, i],
            xaxis=f'x{i + 1}',
            yaxis=f'y{i + 1}',
            line={
                'color': ('rgb(0, 0, 255)'),
                'width': 0.7
            }
        ) for i in range(n_channels)
    ]
    if use_gradient:
        for i, trace in enumerate(traces):
            trace.line['color'] = \
                f'rgb({255*i/n_channels}, 0, {255*(1-i/n_channels)})'
    logger.info("Trace created")
    return traces


def create_annotations(data, n_channels):
    logger.info("Creating annotations")
    return [
        Annotation(
            x=-0.06,
            y=data[:, i].mean(),
            xref='paper',
            yref=f'y{i + 1}',
            text=f'{i + 1}',
            # font=Font(size=9),
            showarrow=False
        ) for i in np.arange(n_channels)
    ]


def get_figure1(data, datetimes):
    logger.info("Setting up figure1")
    return setup_figure(
        data[:, :, 0],
        datetimes,
        use_gradient=False,
        show_yaxis_ticks=False,
        height=900
    )


def get_figure2(data, datetimes, ray):
    logger.info("Setting up figure2")
    return setup_figure(
        data[:, ray, 0].reshape(-1, 1),
        datetimes,
        use_gradient=False,
        show_yaxis_ticks=True,
        height=200
    )


def get_figure3(data, datetimes, ray):
    logger.info("Setting up figure3")
    return setup_figure(
        data[:, ray, 1:],
        datetimes,
        use_gradient=True,
        show_yaxis_ticks=False,
        height=600
    )


def get_figures(data, datetimes, ray):
    fig1 = get_figure1(data, datetimes)
    fig2 = get_figure2(data, datetimes, ray)
    fig3 = get_figure3(data, datetimes, ray)
    return fig1, fig2, fig3
