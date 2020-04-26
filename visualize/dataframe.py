import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams['figure.figsize'] = [10.5, 8]

logger = logging.getLogger('evolvingniches.visualize.pd')
logger.setLevel(logging.INFO)


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

    logger.debug('Grouping messages')
    generational_spectrum = spectra.groupby('generation').mean().sort_index(level='generation')

    fig, ax = plt.subplots(figsize=(10.5, 8))
    p = ax.pcolormesh(generational_spectrum.T, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(p, ax=ax)

    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Spectrum")
    plt.grid()

    if filename is not None:
        logger.debug('Saving file as {}'.format(filename))
        plt.savefig(filename)

    if view:
        logger.debug('Viewing plot')
        plt.show()

    plt.close()


def plot_run_spectra(spectra, cmap='rainbow', view=False, vmin=None, vmax=None, shape=(5, 5),
                     filename='spectra.svg', title="Communication Spectra for All Runs", base_run=None,
                     numbering=False):
    # if not _plt_check(): return

    logger.debug('Compiling data')
    run_generational_spectrum = spectra.groupby(['run', 'generation'], sort=False).mean()

    fig, axes = plt.subplots(*shape, sharex=True, sharey=True, constrained_layout=True)
    runs = run_generational_spectrum.index.get_level_values('run').unique()
    for run, ax in zip(runs, axes.flat):
        logger.debug('Plotting run number {}'.format(run))
        if base_run is None:
            run_spectrum = run_generational_spectrum.loc[(run, slice(None)), :].sort_index(level='generation')
        else:
            run_spectrum = run_generational_spectrum.loc[((base_run, run), slice(None)), :].sort_index(
                level='generation')
        p = ax.pcolormesh(run_spectrum.T, cmap=cmap, vmin=vmin, vmax=vmax)
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
        if numbering:
            ax.set_title(run)

    fig.colorbar(p, ax=axes[:, -1], location='right')

    fig.suptitle(title)

    if filename is not None:
        logger.debug('Saving file as {}'.format(filename))
        plt.savefig(filename)

    if view:
        logger.debug('Viewing plot')
        plt.show()

    logger.debug('Closing plot')
    plt.close()


def plot_channel_volume_histogram(spectra: pd.DataFrame, view=False, vmin=None, vmax=None,
                                  filename='channel_histogram.svg', title="Channel volumes over multiple runs"):
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, constrained_layout=True)
    bins = np.arange(0, 0.7, step=0.1)

    for i, ax in enumerate(axes.flat):
        vh = spectra.iloc[:, i]
        ax.hist(vh, bins=bins)

    fig.suptitle(title)

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_subspecies_abundances(summary, run=None, species=None, role=None, view=False, filename='speciation.svg',
                               sharex=False):
    """Visualizes speciation throughout evolution.

    :param pd.DataFrame counts: The dataframe produced by subspecies_averages_and_counts.
    :param view:
    :param filename:
    :return:
    """

    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    xs_keys = [run, species, role]
    xs_levels = ['run', 'species', 'role']

    xs_keys, xs_levels = list(zip(*filter(lambda f: f[0] is not None, zip(xs_keys, xs_levels))))

    if len(xs_keys) > 0:
        summary = summary.xs(xs_keys, level=xs_levels)

    counts = summary.loc[:, 'counts'].unstack('subspecies')

    fig, ax = plt.subplots()
    counts.plot.area(legend=False, ax=ax)

    ax.set_title("Speciation")
    ax.set_ylabel("Size per Species")
    ax.set_xlabel("Generations")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_subspecies_counts(summary, run=None, species=None, role=None, view=False, filename='speciation.svg'):
    """Visualizes speciation throughout evolution.

    :param pd.DataFrame summary: The dataframe produced by subspecies_averages_and_counts.
    :param view:
    :param filename:
    :return:
    """

    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    counts = summary['counts'].unstack('subspecies').count(axis=1).rename('counts')

    sns.relplot(x='generation', y='counts', hue='role', kind='line', data=counts.reset_index())

    ax = plt.gca()
    ax.set_title("Fitness by Subspecies")
    ax.set_ylabel("Subspecies fitness weighted by size")
    ax.set_xlabel("Generations")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_subspecies_pairplot(summary, run=None, species=None, role=None, view=False, filename='speciation.svg'):
    """Visualizes speciation throughout evolution.

    :param pd.DataFrame summary: The dataframe produced by subspecies_averages_and_counts.
    :param view:
    :param filename:
    :return:
    """

    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    sns.pairplot(data=summary.xs('sender', level='role'), plot_kws={'alpha': 0.05})

    sns.pairplot(data=summary.xs('receiver', level='role'), plot_kws={'alpha': 0.05})


def plot_subspecies_fitness(summary, run=None, species=None, role=None, view=False, filename='subspecies_fitness.svg'):
    """Visualizes the total species fitness, with contributions broken out by subspecies.

    :param pd.DataFrame counts: The dataframe produced by subspecies_averages_and_counts.
    :param view:
    :param filename:
    :return:
    """

    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    xs_keys = [run, species, role]
    xs_levels = ['run', 'species', 'role']

    xs_keys, xs_levels = list(zip(*filter(lambda f: f[0] is not None, zip(xs_keys, xs_levels))))

    if len(xs_keys) > 0:
        summary = summary.xs(xs_keys, level=xs_levels)

    weighted_fitness = summary.loc[:, 'fitness'] * summary.loc[:, 'counts']
    normalized_fitness = weighted_fitness.div(summary.groupby('generation').sum()['counts'])
    fitness = normalized_fitness.unstack('subspecies')

    ax = fitness.plot.area(legend=False)
    ax.set_title("Average fitness over all runs")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generations")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_subspecies_average_fitness(individuals: pd.DataFrame, run=None, species=None, role=None, view=False,
                                    filename='subspecies_fitness.svg'):
    """Visualizes the comparative average fitness of subspecies

    :param pd.DataFrame counts: The dataframe produced by subspecies_averages_and_counts.
    :param view:
    :param filename:
    :return:
    """

    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    xs_keys = [run, species, role]
    xs_levels = ['run', 'species', 'role']

    xs_keys, xs_levels = list(zip(*filter(lambda f: f[0] is not None, zip(xs_keys, xs_levels))))

    if len(xs_keys) > 0:
        individuals = individuals.xs(xs_keys, level=xs_levels)

    fitness = individuals.loc[:, 'fitness']

    sns.lineplot(x="generation", y="fitness",
                 hue="subspecies",
                 data=fitness.reset_index(), palette="Set3")
    # ax = fitness.plot.area(legend=False)
    ax = plt.gca()
    ax.set_title("Fitness by Subspecies")
    ax.set_ylabel("Subspecies fitness weighted by size")
    ax.set_xlabel("Generations")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_species_fitness(summary: pd.DataFrame, run=None, species=None, role=None, view=False, filename='fitness.svg'):
    """Visualizes fitness levels for species

    :param pd.DataFrame summary: The dataframe produced by subspecies_averages_and_counts.
    :param view:
    :param filename:
    :return:
    """

    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    stacked = summary.drop(columns='std').stack()
    stacked.rename('stat', level=-1, inplace=True)
    stacked.name = 'fitness'

    sns.lineplot(x="generation", y="fitness", style="stat",
                 data=stacked.reset_index())

    ax = plt.gca()
    ax.set_title("Fitness by Subspecies")
    ax.set_ylabel("Subspecies fitness weighted by size")
    ax.set_xlabel("Generations")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()
