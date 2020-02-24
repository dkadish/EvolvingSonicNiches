import os

import numpy as np

from analysis.data import load_subset_fn
from visualize import plot_spectrum


def plot_spectrogram_from_multiple_runs(args):
    def load(d):
        species_no = next(iter(d['message_spectra']))
        total = d['message_spectra'][species_no]['total']

        return total

    dataset = load_subset_fn(load, args.dir)

    avg_spectrum = np.average(dataset, axis=0)
    print(avg_spectrum.shape)

    plot_spectrum(avg_spectrum,view=True)


if __name__ == '__main__':
    import argparse

    local_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='Run evolution with a single species.')

    parser.add_argument('dir', metavar='DIR', type=str, action='store',
                        help='base directory for the joblib files')

    args = parser.parse_args()
    print(args)
    plot_spectrogram_from_multiple_runs(args)
