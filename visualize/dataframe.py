import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger('evolvingniches.visualize.pd')

def calculate_spectrum(archive: pd.DataFrame):
    generation_mean = archive.groupby(by=['run', 'generation', 'species']).mean()

    encoded = generation_mean.loc[:, 'encoded_0':'encoded_8']
    received = generation_mean.loc[:, 'received_0':'received_8']

    return encoded.to_numpy(), received.to_numpy()

def plot_spectrum(spectra, cmap='rainbow', view=False, vmin=None, vmax=None,
                  filename='spectrum.svg', title="Use of the communication spectrum by generation"):
    """ """
    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    spectra = np.array(spectra).T
    fig, ax = plt.subplots()
    p = ax.pcolormesh(spectra, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(p, ax=ax)

    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Spectrum")
    plt.grid()

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def calculate_species_counts(individuals: pd.DataFrame):
    subspecies_counts_by_generation = pd.pivot_table(individuals, values='id', index=['run','generation','species','role'], columns='subspecies', aggfunc=np.size, fill_value=0)

    return subspecies_counts_by_generation

def plot_subspecies(counts, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    num_generations = counts.shape[0]
    curves = counts.T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()