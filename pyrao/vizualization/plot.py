"""Contains functions for plotting BSAData."""


def plot_specgram(data, by, limits, resolution, *args, **kwargs):
    """
    Plot spectrogram by frequency bands or by beams.

    Parameters
    ----------
    data : BSAData
        BSAData instance with at least read data in it.
    by : {'freqs', 'beams'}
        Whether to plot specgram by frequencies or by beams.
    limits : array_like
        Start and stop indices of data or start and stop datetimes.
    resolution : float
        Time resolution in seconds.
    *args, **kwargs :
        Extra arguments for plotting function (for example `savefig` etc.)

    """
    raise NotImplementedError


"""
# from matplotlib import pyplot as plt
# from matplotlib.collections import LineCollection

def plot(BSAData, start=0, stop=None, time_resolution='original', line=0,
         kind='freq', plot_type="line_spectre", **kwargs):
    if time_resolution == 'original':
        binsize = 1
    elif time_resolution == 's':
        binsize = int(1//self.resolution)
    elif time_resolution == 'm':
        binsize = int(60//self.resolution)
    elif time_resolution == 'h':
        binsize = int(3600//self.resolution)
    else:
        raise ValueError("time_resolution must be one of the following: \
                          'original', 's', 'm', 'h'")

    stop = stop if stop else len(self.data)
    nsamples = (stop - start) // binsize

    if kind == 'freq':
        y = self.data[start:stop, line, :-1]
        y_labels = self.fbands
        scale = 0.7
    elif kind == 'beam':
        y = self.data[start:stop, :, line]
        y_labels = self.dej
        scale = 0.1
    else:
        raise ValueError("kind must be one of the following: \
                          'freq', 'beam'")

    y = np.array([np.mean(y[i*binsize:(i+1)*binsize], axis=0)
                  for i in range(nsamples)])
    x = self._lin_dt(self.dt[start],
                     self.dt[stop - 1],
                     nsamples)

    if plot_type == 'line_spectre':
        self.plot_line_spectre(x, y, y_labels, scale, **kwargs)

    # plt.plot(x, y, **kwargs)
    # plt.pcolormesh(x, freqs, y.T)

    # fig = plt.figure()
    # ax = fig.gca(projection='2d')
    # plt.plot(np.linspace(1,
    #                      1000,
    #                      (stop-start) // binsize),
    #          [float(freqs[i])] * ((stop-start) // binsize),
    #          y[:,i],
    #          linewidth=0.5)

def plot_line_spectre(BSAData, x, y, y_labels, scale=0.7, figsize=(12, 12),
                      savefig=False, x_labels_format="%H:%M:%S.%f"):
    x_labels = [pd.Timestamp(i)
                  .to_pydatetime()
                  .strftime(x_labels_format)
                for i in x]

    nsamples = y.shape[0]
    nlines = y.shape[1]
    dmin = y.min()
    dmax = y.max()
    dr = (dmax - dmin) * scale

    lines = []
    xticks = []
    for i in range(nlines):
        lines.append(np.array(list(zip(np.linspace(0, 1, nsamples),
                                       y[:, i]))))
        xticks.append(i * dr)

    offsets = np.zeros((nlines, 2), dtype=float)
    offsets[:, 1] = xticks

    ax = plt.figure(figsize=figsize).gca()

    ax.set_ylim(dmin, dmax + dr * (nlines - 1))

    yticks = np.linspace(dmin, dmax + dr * (nlines - 1), nlines + 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([""] + y_labels)
    ax.set_xticklabels(x_labels)
    plt.xticks(rotation=45)

    linecol = LineCollection(lines,
                             offsets=offsets,
                             linewidth=0.5,
                             linestyles='dashed')
    ax.add_collection(linecol)

    if savefig:
        if type(savefig) == bool:
            savefig = './line spectre.png'
        plt.savefig(savefig)

    plt.show()
"""
