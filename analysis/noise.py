import logging
import os
import sys
from datetime import datetime
from itertools import combinations

import numpy as np
import networkx as nx
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn import cluster

en_path =  os.path.abspath(os.path.join(__file__,'..','..'))
print(en_path)
sys.path.append(en_path)

from analysis.data import load_subset_fn
from visualize import plot_spectrum, plot_stats, plot_species, plot_summary_stats

logger = logging.getLogger('analysis.noise')

def __load_all_messages(d):
    """Load all of the messages sent form each run.
    """
    species_no = next(iter(d['messages']['encoded']))
    messages = d['messages']['encoded'][species_no]

    return messages

def __load_messages(d):
    """Load the spectra of messages from each run.

    :param d:
    :return:
    """
    species_no = next(iter(d['message_spectra']))
    total = d['message_spectra'][species_no]['total']

    return total

def calculage_usage(spectrum, generations=50, level=0.15):
    total_generations = spectrum.shape[0]
    channels = spectrum.shape[1]
    extended = np.concatenate([np.zeros(shape=(generations-1, channels)), spectrum])
    usage_map = []
    for i in range(total_generations):
        usage_map.append(np.sum(extended[i:i + generations, :], axis=0) > level * np.sum(extended[i:i + generations, :]))

    usage_map = np.array(usage_map, dtype=float)
    return usage_map

def plot_channel_usage(args):
    """Plot how many individuals are 'using' each channel.
    Usage is defined as more than 20% of the message strength in the last 50 generations.
    """
    messages = load_subset_fn(__load_messages, args.dir)
    usage_maps = []

    for i, ind in enumerate(messages):
        usage_map = calculage_usage(ind)
        usage_maps.append(usage_map)

        number = args.number is None and len(messages) or args.number
        if args.individual and i < number:
            plot_spectrum(usage_map, view=True, title='Usage map (#{})'.format(i), filename='channel_usage_{}.svg'.format(i))

    usage = np.average(usage_maps, axis=0)

    plot_spectrum(usage, view=True, title='Usage map', filename='channel_usage.svg')

def plot_fitness_from_multiple_runs(args):
    view=True
    now = datetime.now()

    def load(d):
        species_no = next(iter(d['encoder_stats']))
        encoder_stats, decoder_stats = d['encoder_stats'][species_no], d['decoder_stats'][species_no]

        return (encoder_stats, decoder_stats)

    stats = load_subset_fn(load, args.dir)

    if args.individual:
        n = args.number is None and len(stats) or args.number
        for i, (encoder_stats, decoder_stats) in enumerate(stats):
            if i >= n: break
            plot_stats(encoder_stats, view=view,
                       filename='data/%s/%s-avg_fitness_enc.svg' % (
                           args.dir, now.strftime('%y-%m-%d_%H-%M-%S')))
            plot_species(encoder_stats, view=view,
                         filename='data/%s/%s-speciation_enc.svg' % (
                             args.dir, now.strftime('%y-%m-%d_%H-%M-%S')))
            plot_stats(decoder_stats, view=view,
                       filename='data/%s/%s-avg_fitness_dec.svg' % (
                           args.dir, now.strftime('%y-%m-%d_%H-%M-%S')))
            plot_species(decoder_stats, view=view,
                         filename='data/%s/%s-speciation_dec.svg' % (
                             args.dir, now.strftime('%y-%m-%d_%H-%M-%S')))

    # Calculate average change between generations
    encoders, decoders = zip(*stats)


    for population in (encoders, decoders):
        best = map(lambda i: [g.fitness for g in i.most_fit_genomes], population)
        avg = map(lambda i: i.get_fitness_mean(), population)
        std = map(lambda i: i.get_fitness_stdev(), population)

        # Average over all runs
        best, avg = [np.average(list(stat), axis=0) for stat in [best, avg]]
        std = np.sum(np.power(list(std), 2))/len(list(std))

        plot_summary_stats(avg, std, best, title='Fitness averages over N runs', xlabel='Generation', ylabel='Fitness', view=True, filename='')

def plot_spectrogram_from_multiple_runs(args):

    spectra = load_subset_fn(__load_messages, args.dir)

    #TODO: double chekc that teh average works
    #TODO: Check the random seed
    #TODO: Include maximums in fitness plot
    #TODO: Noise channel set to 1, they should move away very quickly. We want to know whether they can communicate at all.
            # Possibly add noise to all channels

    if args.individual:
        n = args.number is None and len(spectra) or args.number
        for i, data in enumerate(spectra):
            if i >= n: break
            filename='spectrum.svg'
            if args.save:
                filename = 'spectrum-{}.svg'.format(i)
            plot_spectrum(data, view=True, title='Sender spectrum #{}'.format(i), filename=filename)

    avg_spectrum = np.average(spectra, axis=0)
    print(avg_spectrum.shape)

    plot_spectrum(avg_spectrum,view=True, title='Sender spectrum (total)')

def plot_histogram_of_channel_volume_old(args):
    """ Print a series of 9 histograms showing the distribution of values at a generation of the simulation
    over the N runs for each channel.

    :param args:
    :return:
    """
    generation = args.generation
    n_average = args.n_average

    spectra = load_subset_fn(__load_messages, args.dir)
    run_volumes = []
    for spectrum in spectra:
        if generation is None:
            used_spectrum = spectrum[-n_average:]
        else:
            used_spectrum = spectrum[generation - n_average:generation]

        avg_volume = np.average(used_spectrum, axis=0)
        run_volumes.append(avg_volume)

    volumes = np.concatenate(run_volumes)
    volume_hists = np.hsplit(volumes, 9)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, tight_layout=True)
    bins = np.arange(0, 1, step=0.1)
    # zeros = []

    for i, ax in enumerate(axes.flat):
        vh = volume_hists[i]
        ax.hist(vh, bins=bins)
        # ax.hist(vh[vh>0.01])
        # zeros.append(vh[vh<=0.01].size)

    plt.savefig('spectra_hist.png')
    plt.show()
    plt.close()

    # plt.bar(np.arange(9), zeros)
    # plt.show()

    print('Finished plotting')
    logger.info('Finished Plotting')

def plot_histogram_of_channel_volume(args):
    """ Print a series of 9 histograms showing the distribution of values at a generation of the simulation
    over the N runs for each channel.

    :param args:
    :return:
    """
    generation = args.generation
    n_average = args.n_average

    message_archive = load_subset_fn(__load_all_messages, args.dir)
    run_volumes = []
    for messages_by_run in message_archive:
        if generation is None:
            used_messages = messages_by_run[-n_average:]
        else:
            used_messages = messages_by_run[generation - n_average:generation]

        avg_volume = np.average(np.concatenate(used_messages), axis=0)
        run_volumes.append(avg_volume)

    volumes = np.stack(run_volumes)
    assert volumes.shape == (len(message_archive), 9)
    volume_hists = np.hsplit(volumes, 9)
    volume_hists = list(map(lambda a: np.ravel(a), volume_hists))

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, tight_layout=True)
    bins = np.arange(0,0.7,step=0.1)
    zeros = []

    for i, ax in enumerate(axes.flat):
        vh = volume_hists[i]
        ax.hist(vh, bins=bins)
        # zeros.append(vh[vh<=0.01].size)

    plt.savefig('spectra_hist_all_msgs.png')
    plt.show()
    plt.close()

    # plt.bar(np.arange(9), zeros)
    # plt.show()

    print('Finished plotting')
    logger.info('Finished Plotting')

    # Kolmogorov-Smirnov Test
    # Based on https://towardsdatascience.com/kolmogorov-smirnov-test-84c92fb4158d
    ks_stat = np.zeros(shape=(9, 9))
    ks_p = np.zeros(shape=(9, 9))
    for i, j in combinations(range(9), 2):
        print(i, j)
        ks_calc = ks_2samp(volume_hists[i], volume_hists[j])
        ks_stat[i, j] = ks_calc.statistic
        ks_stat[j, i] = ks_calc.statistic
        ks_p[i, j] = ks_calc.pvalue
        ks_p[j, i] = ks_calc.pvalue

    np.set_printoptions(precision=4, suppress=True)

    # Clustering
    cluster_centers_indices, labels = cluster.affinity_propagation(ks_stat)
    n_clusters_ = len(cluster_centers_indices)

    print('Stat')
    print('Estimated number of clusters: %d' % n_clusters_)
    print(cluster_centers_indices)
    print(labels)

    print(ks_stat)

    print('P-value') # This is the important one? Confidence that the null hypothesis can (low) or cannot (high) be rejected

    cluster_centers_indices, labels = cluster.affinity_propagation(ks_p)
    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    print(cluster_centers_indices)
    print(labels)
    print(ks_p)

    G = nx.Graph()
    G.add_nodes_from(range(9))
    for i in range(9):
        for j in range(i, 9):
            if ks_p[i,j] > 0.01:
                G.add_edge(i, j, weight=ks_stat[i, j])

    plt.plot()
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.savefig('channel_volume_histogram.png')
    plt.show()
    plt.close()

def plot_histogram_of_channel_volume_across_generations(args):
    """Analyze the difference between volume histograms before and after a disturbance

    :param args:
    :return:
    """
    generation = args.generation
    n_average = args.n_average

    message_archive = load_subset_fn(__load_all_messages, args.dir)
    run_volumes_before = []
    run_volumes_after = []
    for messages_by_run in message_archive:
        used_messages_after = messages_by_run[-n_average:]
        used_messages_before = messages_by_run[generation - n_average:generation]

        avg_volume_before = np.average(np.concatenate(used_messages_before), axis=0)
        run_volumes_before.append(avg_volume_before)

        avg_volume_after = np.average(np.concatenate(used_messages_after), axis=0)
        run_volumes_after.append(avg_volume_after)

    volumes_b = np.stack(run_volumes_before)
    assert volumes_b.shape == (len(message_archive), 9)
    volume_b_hists = np.hsplit(volumes_b, 9)
    volume_b_hists = list(map(lambda a: np.ravel(a), volume_b_hists))

    volumes_a = np.stack(run_volumes_after)
    assert volumes_a.shape == (len(message_archive), 9)
    volume_a_hists = np.hsplit(volumes_a, 9)
    volume_a_hists = list(map(lambda a: np.ravel(a), volume_a_hists))

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, tight_layout=True)
    bins = np.arange(0,0.7,step=0.05)
    zeros = []

    for i, ax in enumerate(axes.flat):
        vh_a = volume_a_hists[i]
        vh_b = volume_b_hists[i]
        ax.hist(vh_b, bins=bins, alpha=0.5, label='before')
        ax.hist(vh_a, bins=bins, alpha=0.5, label='after')
        # zeros.append(vh[vh<=0.01].size)

    plt.legend(loc='upper right')
    plt.savefig('spectra_hist_all_msgs.png')
    plt.show()
    plt.close()

    # plt.bar(np.arange(9), zeros)
    # plt.show()

    print('Finished plotting')
    logger.info('Finished Plotting')

    for v, volume_hists in enumerate([volume_a_hists, volume_b_hists]):
        print('GENERATION: {}'.format(v == 0 and generation or 'Final'))

        # Kolmogorov-Smirnov Test
        ks_stat = np.zeros(shape=(9, 9))
        ks_p = np.zeros(shape=(9, 9))
        for i, j in combinations(range(9), 2):
            print(i, j)
            ks_calc = ks_2samp(volume_hists[i], volume_hists[j])
            ks_stat[i, j] = ks_calc.statistic
            ks_stat[j, i] = ks_calc.statistic
            ks_p[i, j] = ks_calc.pvalue
            ks_p[j, i] = ks_calc.pvalue

        np.set_printoptions(precision=4, suppress=True)

        # Clustering
        cluster_centers_indices, labels = cluster.affinity_propagation(ks_stat)
        n_clusters_ = len(cluster_centers_indices)

        print('Stat')
        print('Estimated number of clusters: %d' % n_clusters_)
        print(cluster_centers_indices)
        print(labels)

        print(ks_stat)

        print('P-value') # This is the important one? Confidence that the null hypothesis can (low) or cannot (high) be rejected

        cluster_centers_indices, labels = cluster.affinity_propagation(ks_p)
        n_clusters_ = len(cluster_centers_indices)

        print('Estimated number of clusters: %d' % n_clusters_)
        print(cluster_centers_indices)
        print(labels)
        print(ks_p)

        G = nx.Graph()
        G.add_nodes_from(range(9))
        for i in range(9):
            for j in range(i, 9):
                if ks_p[i,j] > 0.01:
                    G.add_edge(i, j, weight=ks_stat[i, j])

        plt.plot()
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.savefig('channel_volume_histogram_{}.png'.format(v))
        plt.show()
        plt.close()

    # Kolmogorov-Smirnov Test For Generations G and Final
    print('Channel: KS-Value, P-value')
    ks_stat = []
    ks_p = []
    for i in range(9):
        ks_calc = ks_2samp(volume_a_hists[i], volume_b_hists[i])
        ks_stat.append(ks_calc.statistic)
        ks_p.append(ks_calc.pvalue)
        print('{}: {}, {}'.format(i, ks_calc.statistic, ks_calc.pvalue))

    fig, axes = plt.subplots(2, 1, sharex=True, tight_layout=True)
    axes[0].bar(range(9), ks_stat)
    axes[1].bar(range(9), ks_p)
    plt.savefig('before_after_ks.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    import argparse


    d = input("dir: ")
    subparser = input("subparser: ")

    local_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='Run evolution with a single species.')

    parser.add_argument('dir', metavar='DIR', type=str, action='store',
                        help='base directory for the joblib files')
    parser.add_argument('-n','--number', metavar='N', type=int, default=None,
                        help='only plot the first N runs')
    parser.add_argument('-i','--individual', action='store_true', default=False,
                        help='plot individual plots as well as the summary plot')
    parser.add_argument('-s','--save', action='store_true', default=False,
                        help='save the plots to a folder')

    subparsers = parser.add_subparsers()
    parser_spectrum = subparsers.add_parser('spectrum')
    parser_spectrum.set_defaults(func=plot_spectrogram_from_multiple_runs)

    parser_scores = subparsers.add_parser('scores')
    parser_scores.set_defaults(func=plot_fitness_from_multiple_runs)

    parser_usage = subparsers.add_parser('usage')
    parser_usage.set_defaults(func=plot_channel_usage)

    parser_hist = subparsers.add_parser('hist')
    parser_hist.set_defaults(func=plot_histogram_of_channel_volume)
    parser_hist.add_argument('-g','--generation', metavar='G', type=int, default=None,
                        help='Which generation to plot at')
    parser_hist.add_argument('--n-average', metavar='N', type=int, default=10,
                        help='Average over N generations')

    parser_hist_gen = subparsers.add_parser('hist_gen')
    parser_hist_gen.set_defaults(func=plot_histogram_of_channel_volume_across_generations)
    parser_hist_gen.add_argument('-g','--generation', metavar='G', type=int, default=299,
                        help='Which generation to plot at')
    parser_hist_gen.add_argument('--n-average', metavar='N', type=int, default=10,
                        help='Average over N generations')

    a = sys.argv
    print(sys.argv)
    arg_array = [d, ] + sys.argv[1:] + [subparser, ]

    print(str(arg_array))
    args = parser.parse_args(arg_array)
    print(args)
    args.func(args)
    # plot_spectrogram_from_multiple_runs(args)
