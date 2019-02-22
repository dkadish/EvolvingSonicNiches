import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.manifold import TSNE


def _plt_unavailable_warning():
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return


def _finish(filename, fmt, view):
    # Save and show
    plt.savefig('{}.{}'.format(filename, fmt), bbox_inches='tight')
    if view:
        plt.show()

    plt.close()


def generations_of_interest(silhouette):
    _prominence = 0.2
    peaks = np.concatenate([find_peaks(silhouette, prominence=_prominence)[0], find_peaks(1 - silhouette, prominence=_prominence)[0],
                            [0, silhouette.shape[-1] - 1]])
    return np.array(sorted(peaks))


def plot_silhouette(silhouette, null=None, cmap_name='PuOr', goi=False, individual=None, aspect=2, view=False,
                    filename='silhouette', fmt='pdf'):
    _plt_unavailable_warning()

    # Prepare the data
    generation = np.array(range(silhouette.shape[1]))
    averages = np.average(silhouette, axis=0)
    stds = np.std(silhouette, axis=0)

    # Prepare the colourmap
    cmap = plt.cm.get_cmap(cmap_name, lut=7)

    # Plot the main data
    _size = 6
    fig, ax = plt.subplots(figsize=(_size, _size / aspect))
    ax.plot(generation, averages, linewidth=1, label='avg', c=[0,0,0,0.6])#cmap(1 / 7))
    ax.fill_between(generation, averages - stds, averages + stds, facecolor=[0,0,0,0.2])#cmap(1 / 7), alpha=0.25)

    # Plot an individual sample
    if individual is not None:
        c = [208/255.0, 28/255.0, 139/255.0, 0.6]#cmap(6/7)
        # c = c[:-1] + (0.5,)
        ax.plot(generation, silhouette[individual, :], label='example', linewidth=1, color=c)

        if goi:
            generations = generations_of_interest(silhouette[individual, :])
            ax.scatter(generation[generations], silhouette[individual, generations], label='plotted', color=c)

    if null is not None:
        averages_n = np.average(null, axis=0)
        stds_n = np.std(null, axis=0)
        ax.plot(generation, averages_n, linewidth=0.5, label='null avg', c=[0, 0, 0, 0.3])#cmap(4 / 7))
        ax.fill_between(generation, averages_n - stds_n, averages_n + stds_n, facecolor=[0, 0, 0, 0.05])#cmap(4 / 7), alpha=0.25)


    # Text
    # fig.suptitle('')
    ax.tick_params(labelsize='xx-small')
    plt.xlabel('Generation', fontsize='small')
    plt.ylabel('Silhouette Score', fontsize='small')
    # plt.grid()
    plt.legend(loc="best", fontsize='x-small')

    _finish(filename, fmt, view)


def plot_clusters(messages, generations=None, cmap_name='RdBu', individual=None, aspect=2, view=False,
                  filename='clusters',
                  fmt='pdf'):
    '''

    :param silhouette:
    :param messages: { 'group': [ generation 0 (np.array), ..., generation N], 'encoded': }
    :param cmap_name:
    :param individual:
    :param aspect:
    :param view:
    :param filename:
    :param fmt:
    :return:
    '''
    _plt_unavailable_warning()

    if individual is None:
        warnings.warn("An individual run must be selected for this plot")
        return

    if generations is None:
        generations = list(range(0, 300, 100)) + [299]
    group = messages['group']
    encoded = messages['encoded']

    n_plots = len(generations)

    two_d = dict([(g, TSNE(n_components=2).fit_transform(encoded[g])) for g in generations])
    min_2d = 1.5 * np.min([np.min(two_d[g], axis=0) for g in generations], axis=0)
    max_2d = 1.5 * np.max([np.max(two_d[g], axis=0) for g in generations], axis=0)
    min_x, min_y, max_x, max_y = min_2d[0], min_2d[1], max_2d[0], max_2d[1]

    # Plot parameters
    _size = 6
    fig, ax = plt.subplots(nrows=1, ncols=n_plots, figsize=(_size, _size / aspect))

    for i, g in enumerate(generations):
        ax = plt.subplot2grid((1, n_plots), (0, i))
        # Prepare the colourmap
        cmap = plt.cm.get_cmap(cmap_name, lut=7)
        colours = cmap([0.25, 0.75])
        colours[:, -1] = 0.25  # Set alpha

        # TODO This doesn't work for ALL and labels is still broken.
        indices = [group[g] == b for b in set(group[g])]

        # TODO Try the legand((LABELS),(STUFF)... format
        axes = []

        two_d = TSNE(n_components=2).fit_transform(encoded[g])

        for b, c in zip(set(group[g]), colours):
            index = group[g] == b
            axes.append(ax.scatter(two_d[index, 0], two_d[index, 1], s=5,
                                   edgecolors='none', color=c))

        ax.set_title(g)
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.set_aspect('equal', share=True)
        ax.set_xlim(left=min_x, right=max_x)
        ax.set_ylim(bottom=min_y, top=max_y)

        # if g == generations[-1]:
        #     ax.legend(axes, [b for b in set(group[g])], bbox_to_anchor=(1.05, 1),
        #               borderaxespad=0., loc='upper left', fontsize='small', fancybox=True)

    fig.tight_layout()

    _finish(filename, fmt, view)


def plot_scores(cmap_name='PuOr', individual=None, aspect=2, view=False, filename='scores', fmt='pdf'):
    pass


def plot_fitness(cmap_name='PuOr', individual=None, aspect=2, view=False, filename='fitness', fmt='pdf'):
    pass


def _get_species(data):
    return list(data['scores'].keys())


def _do_plot_all(data, null=None, individual=0):
    _do_plot_silhouette(data, null, individual, goi=True)
    _do_plot_clusters(data, null, individual)


def _do_plot_silhouette(data, null=None, individual=None, goi=False):
    silhouette = list(map(lambda d: d['silhouette']['ALL'], data))
    silhouette = np.array(silhouette)

    silhouette_n = None
    if null is not None:
        silhouette_n = list(map(lambda d: d['silhouette']['ALL'], null))
        silhouette_n = np.array(silhouette_n)

    plot_silhouette(silhouette, null=silhouette_n, goi=goi, individual=individual, view=True)


def _do_plot_clusters(data, null=None, individual=0):
    silhouette = list(map(lambda d: d['silhouette']['ALL'], data))
    silhouette = np.array(silhouette[individual])

    peaks = generations_of_interest(silhouette)

    messages = {
        'group': [],
        'encoded': []
    }

    encoded = data[0]['messages']['encoded']

    for generation in zip(*[encoded[e] for e in sorted(encoded.keys())]):
        messages['encoded'].append(np.append(*generation, axis=0))
        o = [i * np.ones(g.shape[0]) for i, g in enumerate(generation)]
        messages['group'].append(np.append(*o, axis=0))

    plot_clusters(messages, generations=peaks, individual=individual, view=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test graphing functions.')
    parser.add_argument('dir', help='Folder to search within for the data files.')
    parser.add_argument('datafile', help='File containing joblibed run data.')
    parser.add_argument('--null', help='Folder containing joblibed run data for the null case.', default=None)

    subparsers = parser.add_subparsers(help='sub-command help')

    all_parser = subparsers.add_parser('all', help='Plot all plots.')
    all_parser.set_defaults(func=_do_plot_all)

    silhouette_parser = subparsers.add_parser('silhouette', help='Plot silhouette scores.')
    silhouette_parser.set_defaults(func=_do_plot_silhouette)

    clusters_parser = subparsers.add_parser('cluster', help='Test Cluster plotting')
    clusters_parser.set_defaults(func=_do_plot_clusters)

    arguments = parser.parse_args()

    datafiles = []
    # Load data files
    for root, dirs, files in os.walk(arguments.dir):
        for file in files:
            if file.strip() == arguments.datafile.strip():
                datafiles.append(os.path.join(root, file))

    data = []
    for f in datafiles:
        data.append(joblib.load(f))

    if arguments.null is not None:
        datafiles = []
        # Load data files
        for root, dirs, files in os.walk(arguments.null):
            for file in files:
                if file.strip() == arguments.datafile.strip():
                    datafiles.append(os.path.join(root, file))

        null = []
        for f in datafiles:
            null.append(joblib.load(f))
    else:
        null = None

    arguments.func(data, null)

# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# from scipy.signal import find_peaks
# import joblib
# data = joblib.load('./runs/data/19B20_0200 - 5 runs/19B20_0209/data.joblib')
# generations = np.array(range(300))
# silhouette = data['silhouette']['ALL']
# silhouette = np.array(silhouette)
# peaks = np.concatenate([find_peaks(silhouette, prominence=0.15)[0], find_peaks(1-silhouette, prominence=0.15)[0], [generations[0], generations[-1]]])
