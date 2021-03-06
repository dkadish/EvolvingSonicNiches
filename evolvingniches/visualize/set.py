import os
import warnings
from string import ascii_uppercase

import humanize
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE

INDIVIDUAL = 5


def _plt_unavailable_warning():
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return


def _finish(filename, fmt, view):
    # Save and show
    plt.savefig('{}.{}'.format(filename, fmt), bbox_inches='tight', dpi=1200)
    if view:
        plt.show()

    plt.close()


def generations_of_interest(silhouette):
    _prominence = 0.175
    peaks = np.concatenate(
        [find_peaks(silhouette, prominence=_prominence)[0], find_peaks(1 - silhouette, prominence=_prominence)[0],
         [0, silhouette.shape[-1] - 1]])
    return np.array(sorted(peaks))


def plot_silhouette(silhouette, null=None, cmap_name='PuOr', goi=False, individual=None, aspect=2, view=False,
                    filename='silhouette', fmt='pdf'):
    _plt_unavailable_warning()

    # Prepare the data
    n_alternative = silhouette.shape[0]

    generation = np.array(range(silhouette.shape[1]))
    averages = np.average(silhouette, axis=0)
    # var = np.var(silhouette, axis=0)
    stds = np.std(silhouette, axis=0)

    # Prepare the colourmap
    cmap = plt.cm.get_cmap(cmap_name, lut=7)

    # Plot the main data
    _size = 6
    fig, ax = plt.subplots(figsize=(_size, _size / aspect))
    # fig, (ax, ax_p) = plt.subplots(nrows=2, sharex=True, figsize=(_size, _size / aspect), gridspec_kw={'height_ratios':[2, 1]})
    ax.plot(generation, averages, linewidth=1, label='$H_1 \quad (n={})$'.format(n_alternative),
            c=[0, 0, 0, 0.6])  # cmap(1 / 7))
    ax.fill_between(generation, averages - stds, averages + stds, facecolor=[0, 0, 0, 0.2])  # cmap(1 / 7), alpha=0.25)

    if null is not None:
        n_null = null.shape[0]
        averages_n = np.average(null, axis=0)
        # var_n = np.var(silhouette, axis=0)
        stds_n = np.std(null, axis=0)
        ax.plot(generation, averages_n, linewidth=0.5, label='$H_0 \quad (n={})$'.format(n_null),
                c=[0, 0, 0, 0.3])  # cmap(4 / 7))
        ax.fill_between(generation, averages_n - stds_n, averages_n + stds_n,
                        facecolor=[0, 0, 0, 0.05])  # cmap(4 / 7), alpha=0.25)

        # Statistical testing
        # t = (averages - averages_n)/np.sqrt(np.power(var, 2)/n_alternative + np.power(var_n, 2)/n_null)
        # v = np.power(np.power(var, 2)/n_alternative + np.power(var_n, 2)/n_null, 2) / \
        #     (np.power(var, 4)/(n_alternative**2*(n_alternative-1)) + np.power(var_n, 4)/(n_null**2*(n_null-1)))
        statistic, p = ttest_ind(silhouette, null, axis=0, equal_var=False)
        significant = p < 0.01
        indices = list(np.nonzero(significant[1:] != significant[:-1])[0] + 1)
        first_true = significant[0]

        if first_true:
            indices = [0] + indices
        indices = indices + [significant.size - 1]

        print('Average P-Value (Total): {}'.format(np.average(p)))
        for i, j in zip(indices[:-1:2], indices[1::2]):
            print(i, j)
            print('Average P-Value ({}): {}'.format(j - i, np.average(p[i:j])))
            print('Max P-Value ({}): at Gen {} = {}'.format(j - i, np.argmax(p[i:j]), np.max(p[i:j])))
            # ax.axvspan(i, j, facecolor=[0, 0, 0, 0.1])

        # ax_p.semilogy(generation, p, linewidth=1.0, label='P-value'.format(n_null), c=[1, 0, 0, 0.5])
        # ax_p.semilogy(generation, 0.05*np.ones(shape=p.shape), linewidth=1.0, label='P=0.05'.format(n_null), c=[0, 1, 0, 0.5])
        # ax_p.tick_params(labelsize='xx-small')
        # ax_p.set_ylabel("Significance", fontsize='x-small')
        # ax_p.set_ylim(top=1.0)
        # ax_p.legend(loc="best", fontsize='x-small')

    # Plot an individual sample
    if individual is not None:
        c = [208 / 255.0, 28 / 255.0, 139 / 255.0, 0.6]  # cmap(6/7), #D01C8B
        # c = c[:-1] + (0.5,)
        ax.plot(generation, silhouette[individual, :], label='example', linewidth=1, color=c)

        if goi:
            generations = generations_of_interest(silhouette[individual, :])
            ax.scatter(generation[generations], silhouette[individual, generations], label='plotted', color=c)

    # Text
    # fig.suptitle('')
    ax.tick_params(labelsize='xx-small')
    ax.set_xlabel('Generation', fontsize='small')
    ax.set_ylabel('Silhouette Score', fontsize='small')
    # plt.grid()
    ax.legend(loc="best", fontsize='x-small')

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
        # colours = cmap([0.25, 0.75])
        colours = np.array([[216, 193, 51, 255],
                            [5, 148, 157, 255]]) / 255.0
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


def plot_cluster_video(messages, cmap_name='RdBu', aspect=2, view=False,
                       filename='clusters',
                       fmt='png', _last_generation=300, align_images=False):
    '''

    :param silhouette:
    :param messages: { 'group': [ generation 0 (np.array), ..., generation N], 'encoded': }
    :param cmap_name:
    :param aspect:
    :param view:
    :param filename:
    :param fmt:
    :return:
    '''
    _plt_unavailable_warning()

    generations = list(range(0, _last_generation))
    group = messages['group']
    encoded = messages['encoded']

    n_plots = len(generations)

    # Find the 2d representations
    two_d = dict([(g, TSNE(n_components=2).fit_transform(encoded[g])) for g in generations])

    if align_images:
        # Match the cluster centriods from each cluster, use https://yutsumura.com/find-a-matrix-that-maps-given-vectors-to-given-vectors/
        transformed = {}
        # g = generations[0]
        # mean = np.array([np.mean(two_d[g][group[g] == 0], axis=0), np.mean(two_d[g][group[g] == 1], axis=0)]).T
        mean = np.array([[0.1, 0], [0, 0.1]])

        for g in generations:
            mean0 = np.mean(two_d[g][group[g] == 0], axis=0)
            mean1 = np.mean(two_d[g][group[g] == 1], axis=0)
            X = np.array([mean0, mean1]).T
            # mean = (mean + X) / 2
            Xinv = np.linalg.inv(X)
            A = np.dot(mean, Xinv)

            u, s, vh = np.linalg.svd(A)
            D = np.array([[1, 0], [0, np.linalg.det(np.dot(u, vh))]])
            R = np.dot(u, vh)

            transformed[g] = np.dot(R, two_d[g].T).T
    else:
        transformed = two_d

    min_2d = 1.25 * np.min([np.min(transformed[g], axis=0) for g in generations], axis=0)
    max_2d = 1.25 * np.max([np.max(transformed[g], axis=0) for g in generations], axis=0)
    min_x, min_y, max_x, max_y = min_2d[0], min_2d[1], max_2d[0], max_2d[1]

    # Plot parameters
    _size = 6
    for i, g in enumerate(generations):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(_size, _size / aspect))

        # ax = plt.subplot2grid((1, n_plots), (0, i))
        # Prepare the colourmap
        cmap = plt.cm.get_cmap(cmap_name, lut=7)
        # colours = cmap([0.25, 0.75])
        colours = np.array([[216, 193, 51, 255],
                            [5, 148, 157, 255]]) / 255.0
        colours[:, -1] = 0.25  # Set alpha

        # TODO This doesn't work for ALL and labels is still broken.
        indices = [group[g] == b for b in set(group[g])]

        # TODO Try the legand((LABELS),(STUFF)... format
        axes = []

        for b, c in zip(set(group[g]), colours):
            index = group[g] == b
            axes.append(ax.scatter(transformed[g][index, 0], transformed[g][index, 1], s=5,
                                   edgecolors='none', color=c))

        # min_2d = 1.5 * np.min(two_d[g], axis=0)
        # max_2d = 1.5 * np.max(two_d[g], axis=0)
        # min_x, min_y, max_x, max_y = min_2d[0], min_2d[1], max_2d[0], max_2d[1]

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

        _finish('{}_{:03d}'.format(filename, i), fmt, view)


def plot_scores(scores, cmap_name='PuRd', individual=None, aspect=2, view=False, filename='scores', fmt='pdf'):
    _plt_unavailable_warning()

    # Prepare the colourmap
    cmap = plt.cm.get_cmap(cmap_name, lut=3)

    colours = np.array([[216, 193, 51, 255],
                        [5, 148, 157, 255],
                        [208, 28, 139, 255]]) / 255.0

    # Prepare the data
    avgs = dict([(s, np.average(scores[s], axis=0)) for s in scores])
    stds = dict([(s, np.std(scores[s], axis=0)) for s in scores])
    if individual is not None:
        inds = dict([(s, scores[s][individual]) for s in scores])

    generation = range(avgs['species'].shape[0])

    adj = 3
    den = 6

    for (i, s), c in zip(enumerate(scores), colours):
        plt.plot(generation, avgs[s], c=c, label=s)
        plt.fill_between(generation, avgs[s] - stds[s], avgs[s] + stds[s], facecolor=c, alpha=0.5)

        if individual is not None:
            plt.plot(generation, inds[s], c=c, alpha=0.75, linestyle=':')

    plt.title("Scores by generation")
    plt.xlabel("Generation")
    plt.ylabel("Scores")
    plt.grid()
    plt.gca().set_ylim(bottom=0, top=1)
    plt.legend(loc="best")

    _finish(filename, fmt, view)


def plot_n_channels(channels, null=None, view=False, filename='n_channels', fmt='pdf'):
    _plt_unavailable_warning()

    lower = 10
    upper = 40

    channels = channels[:-1, :, :]

    generations = range(channels.shape[0])
    n_channels = []

    for threshold in range(lower, upper):
        n_channels.append(np.count_nonzero(channels > threshold, axis=1))

    n_channels = np.array(n_channels)
    run_avg = np.average(n_channels, axis=0)
    run_std = np.std(n_channels, axis=0)

    avg = np.average(run_avg, axis=1)
    # std = np.sqrt(np.sum(np.power(run_std,2), axis=1))
    std = np.std(run_avg, axis=1)

    plt.plot(generations, avg)
    plt.fill_between(generations, avg - std, avg + std, alpha=0.2)

    if null is not None:
        channels_null = null[:-1, :, :]
        print(channels_null.shape)

        n_channels_null = []

        for threshold in range(lower, upper):
            n_channels_null.append(np.count_nonzero(channels_null > threshold, axis=1))

        n_channels_null = np.array(n_channels_null)
        run_avg_null = np.average(n_channels_null, axis=0)
        run_std_null = np.std(n_channels_null, axis=0)

        avg_null = np.average(run_avg_null, axis=1)
        # std_null = np.sqrt(np.sum(np.power(run_std_null,2), axis=1))
        std_null = np.std(run_avg_null, axis=1)

        plt.plot(generations, avg_null)
        plt.fill_between(generations, avg_null - std_null, avg_null + std_null, alpha=0.2)

    # # Plot an individual sample
    # if individual is not None:
    #     c = [208 / 255.0, 28 / 255.0, 139 / 255.0, 0.6]  # cmap(6/7)
    #     # c = c[:-1] + (0.5,)
    #     ax.plot(generation, silhouette[individual, :], label='example', linewidth=1, color=c)
    #
    #     if goi:
    #         generations = generations_of_interest(silhouette[individual, :])
    #         ax.scatter(generation[generations], silhouette[individual, generations], label='plots', color=c)

    plt.title("Number of channels used")
    plt.xlabel("Generations")
    plt.ylabel("Channels")
    plt.grid()
    plt.legend(loc="best")

    _finish(filename, fmt, view)


def plot_spectrum(spectra, cmap='rainbow', aspect=3.5, view=False,
                  filename='spectrum', fmt='pdf'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    letters = iter(ascii_uppercase)

    _size = 10
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(_size, _size / aspect))

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # plt.title("Use of the communication spectrum by generation")
    plt.ylabel("Frequency Band")
    # plt.grid()

    for spectrum, ax in zip(spectra, axes.flat):
        s = np.array(spectrum).T
        p = ax.pcolormesh(s[:, :-1], cmap=cmap)
        ax.tick_params(labelsize='xx-small')
        ax.set_yticks(np.arange(0.5, 9.0, 1.0))
        ax.set_yticklabels(np.arange(0, 9, 1))
        ax.set_title('Species {}'.format(next(letters)))
        ax.set_xlabel("Generation")

    fig.tight_layout()
    cb = fig.colorbar(p, ax=axes.flat)
    cb.ax.tick_params(labelsize='x-small')

    _finish(filename, fmt, view)


def plot_fitness(cmap_name='PuOr', individual=None, aspect=2, view=False, filename='fitness', fmt='pdf'):
    pass


def _get_species(data):
    return list(data['scores'].keys())


def _do_plot_all(data, null=None, individual=INDIVIDUAL):
    _do_plot_silhouette(data, null, individual, goi=True)
    _do_plot_clusters(data, null, individual)


def _do_plot_silhouette(data, null=None, individual=INDIVIDUAL, goi=True):
    silhouette = list(map(lambda d: d['silhouette']['ALL'], data))
    silhouette = np.array(silhouette)

    silhouette_n = None
    if null is not None:
        silhouette_n = list(map(lambda d: d['silhouette']['ALL'], null))
        silhouette_n = np.array(silhouette_n)

    plot_silhouette(silhouette, null=silhouette_n, goi=goi, individual=individual, view=True)


def _do_plot_clusters(data, null=None, individual=INDIVIDUAL):
    silhouette = list(map(lambda d: d['silhouette']['ALL'], data))
    silhouette = np.array(silhouette[individual])

    peaks = generations_of_interest(silhouette)

    messages = {
        'group': [],
        'encoded': []
    }

    encoded = data[individual]['messages']['encoded']

    for generation in zip(*[encoded[e] for e in sorted(encoded.keys())]):
        messages['encoded'].append(np.append(*generation, axis=0))
        o = [i * np.ones(g.shape[0]) for i, g in enumerate(generation)]
        messages['group'].append(np.append(*o, axis=0))

    plot_clusters(messages, cmap_name='RdPu', generations=peaks, individual=individual, view=True)


def _do_plot_cluster_video(data, null=None, individual=INDIVIDUAL):
    messages = {
        'group': [],
        'encoded': []
    }

    try:
        encoded = data[individual]['messages']['encoded']
    except IndexError as e:
        print('Index error. No run number {} in the set. Proceeding with first sample.'.format(individual))
        encoded = data[0]['messages']['encoded']

    for generation in zip(*[encoded[e] for e in sorted(encoded.keys())]):
        messages['encoded'].append(np.append(*generation, axis=0))
        o = [i * np.ones(g.shape[0]) for i, g in enumerate(generation)]
        messages['group'].append(np.append(*o, axis=0))

    plot_cluster_video(messages, cmap_name='RdPu', view=True)


def _do_plot_scores(data, null=None, individual=None):
    scores = {}
    for d in data:
        for sp in d['scores']:
            for sc in d['scores'][sp]:
                if sc not in scores:
                    scores[sc] = []
                scores[sc].append(d['scores'][sp][sc])

    for s in scores:
        scores[s] = np.array(scores[s])

    plot_scores(scores, individual=individual)


def _do_plot_n_channels(data, null=None, individual=None):
    channels = []
    for d in data:
        for sp in d['message_spectra']:
            channels.append(d['message_spectra'][sp]['total'])

    channels = np.array(channels)
    channels = np.moveaxis(channels, 0, -1)

    if null is not None:
        channels_n = []
        for d in null:
            for sp in d['message_spectra']:
                channels_n.append(d['message_spectra'][sp]['total'])
        channels_n = np.array(channels_n)
        channels_n = np.moveaxis(channels_n, 0, -1)

    plot_n_channels(channels, null=channels_n)


def _do_plot_spectra_for_run(data, null=None, individual=None):
    letter = iter(ascii_uppercase)

    if individual is None:
        individuals = range(len(data))
    else:
        individuals = [individual]

    for i in individuals:
        d = data[i]

        spectra = []
        for sp in d['message_spectra']:
            spectra.append(d['message_spectra'][sp]['total'])

        plot_spectrum(spectra, cmap='RdPu', filename='spectrum-{}'.format(i))


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

    cluster_video_parser = subparsers.add_parser('cluster_video', help='Cluster Video plotting')
    cluster_video_parser.set_defaults(func=_do_plot_cluster_video)

    scores_parser = subparsers.add_parser('scores', help='Test score plotting')
    scores_parser.set_defaults(func=_do_plot_scores)

    n_channels_parser = subparsers.add_parser('n_channels', help='Test n channel plotting')
    n_channels_parser.set_defaults(func=_do_plot_n_channels)

    spectra_parser = subparsers.add_parser('spectra', help='Plot spectra for a particular run')
    spectra_parser.set_defaults(func=_do_plot_spectra_for_run)

    arguments = parser.parse_args()

    import time

    start = time.time()

    datafiles = []
    # Load data files
    for root, dirs, files in os.walk(arguments.dir):
        for file in files:
            if file.strip() == arguments.datafile.strip():
                print('{}: {}'.format(len(datafiles), root))
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

    end = time.time()
    print(humanize.naturaltime(end - start))

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
