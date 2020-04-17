import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logger = logging.getLogger('evolvingniches.visualize.pd')

def _plt_check():
    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return False

    return True

def plot_spectrum(spectra, cmap='rainbow', view=False, vmin=None, vmax=None,
                  filename='spectrum.svg', title="Average communication spectrum for all runs"):
    """Plots the spectrum from a dataframe produced by dataframe.calculations.spectrum.encoded_by_run_generation or
    dataframe.calculations.spectrum.received_by_run_generation.

    The dataframe can be pre-filtered to only include certain runs, generations, or species.

    :param spectra:
    :param cmap:
    :param view:
    :param vmin:
    :param vmax:
    :param filename:
    :param title:
    :return:
    """
    if not _plt_check(): return

    generational_spectrum = spectra.groupby('generation').mean()

    fig, ax = plt.subplots()
    p = ax.pcolormesh(generational_spectrum.T, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(p, ax=ax)

    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Spectrum")
    plt.grid()

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

def plot_run_spectra(spectra, cmap='rainbow', view=False, vmin=None, vmax=None, shape=(5,5),
                  filename='spectra.svg', title="Communication Spectra for All Runs"):
    # if not _plt_check(): return

    run_generational_spectrum = spectra.groupby(['run', 'generation']).mean()

    fig, axes = plt.subplots(*shape, sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        p = ax.pcolormesh(run_generational_spectrum.loc[(i, slice(None)), :].T, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False)  # labels along the bottom edge are off

    fig.colorbar(p, ax=axes[:,-1], location='right')

    fig.suptitle(title)
    # axes[-1, shape[0]//2].set_xlabel("Generations")
    # axes[shape[1]//2, 0].set_ylabel("Spectrum")
    # plt.grid()

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

def plot_channel_volume_histogram(spectra: pd.DataFrame, view=False, vmin=None, vmax=None,
                  filename='channel_histogram.svg', title="Channel volumes over multiple runs"):

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, constrained_layout=True)
    bins = np.arange(0, 0.7, step=0.1)

    for i, ax in enumerate(axes.flat):
        vh = spectra.iloc[:,i]
        ax.hist(vh, bins=bins)

    fig.suptitle(title)

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()



############### NOT IMPLEMENTED PROPERLY
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