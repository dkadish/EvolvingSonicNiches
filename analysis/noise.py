import os
import sys

import numpy as np
from scipy.signal import savgol_filter

from analysis.data import load_subset_fn
from visualize import plot_spectrum
import matplotlib.pyplot as plt

en_path =  os.path.abspath(os.path.join(__file__,'..','..'))
print(en_path)
sys.path.append(en_path)

def __load_messages(d):
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

        if args.individual and i < args.number:
            plot_spectrum(usage_map, view=True, title='Usage map (#{})'.format(i))

    usage = np.average(usage_maps, axis=0)

    plot_spectrum(usage, view=True, title='Usage map')

def plot_fitness_from_multiple_runs(args):
    def load(d):
        species_no = next(iter(d['scores']))
        total = d['scores'][species_no]['total']

        return total

    scores = load_subset_fn(load, args.dir)

    if args.individual:
        n = args.number is None and len(scores) or args.number
        for i, data in enumerate(scores):
            if i >= n: break
            plt.plot(range(data.size), savgol_filter(data, 11, 3), label=i, alpha=0.5)

            # plot_spectrum(data, view=True, title='Sender spectrum #{}'.format(i))

    # Calculate average change between generations
    s = np.array(scores)
    diffs = s[:, 1:] - s[:, :-1]
    avg_diffs = np.concatenate([[0, ], np.average(diffs, axis=0)])

    avg_scores = np.average(scores, axis=0)
    plt.plot(range(avg_scores.size), savgol_filter(avg_scores, 11, 3), label="tot")

    # ax2 = plt.gca().twinx()
    # ax2.plot(range(avg_scores.size), savgol_filter(avg_diffs, 11, 3), label="dif")
    plt.legend()

    plt.show()

    if args.save:
        plt.savefig('fitness.svg')

    # avg_spectrum = np.average(spectra, axis=0)
    # print(avg_spectrum.shape)
    #
    # plot_spectrum(avg_spectrum,view=True, title='Sender spectrum (total)')

def plot_spectrogram_from_multiple_runs(args):

    spectra = load_subset_fn(__load_messages, args.dir)

    #TODO: double chekc that teh average works
    #TODO: Check the random seed

    if args.individual:
        n = args.number is None and len(spectra) or args.number
        for i, data in enumerate(spectra):
            if i >= n: break
            if args.save:
                filename = 'spectrum-{}.svg'.format(i)
            plot_spectrum(data, view=True, title='Sender spectrum #{}'.format(i), filename=filename)

    avg_spectrum = np.average(spectra, axis=0)
    print(avg_spectrum.shape)

    plot_spectrum(avg_spectrum,view=True, title='Sender spectrum (total)')


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

    parser_scores = subparsers.add_parser('usage')
    parser_scores.set_defaults(func=plot_channel_usage)

    a = sys.argv
    print(sys.argv)
    arg_array = [d, ] + sys.argv[1:] + [subparser, ]

    print(str(arg_array))
    args = parser.parse_args(arg_array)
    print(args)
    args.func(args)
    # plot_spectrogram_from_multiple_runs(args)
