import itertools
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


def _plt_finish(view=False, filename='plot.svg', close=True):
    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    if close:
        plt.close()


def plot_spectrum(spectra, cmap='rainbow', view=False, close=True, vmin=None, vmax=None,
                  filename='spectrum.svg', title="Average communication spectrum for all runs",
                  xlabel='Generation', ylabel='Spectrum',
                  colorbar=True, **kwargs):
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

    figsize = (10.5, 8)
    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = plt.gca()
        fig = ax.get_figure()

    p = ax.pcolormesh(generational_spectrum.T, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.yticks([0.5 + i for i in range(9)], labels=range(9))
    plt.grid(b=None, axis='y')
    if colorbar:
        fig.colorbar(p, ax=ax)

    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.grid()

    _plt_finish(view, filename, close)


def plot_spectra_and_fitness(spectra: pd.DataFrame, species: pd.DataFrame, cmap='rainbow', view=False, close=True,
                             vmin=None, vmax=None, shape=(5, 5),
                             filename='spectra_fitness.svg', title="Communication Spectra and Fitness for All Runs",
                             base_run=None,
                             numbering=False, **kwargs):
    logger.debug('Compiling data')
    run_generational_spectrum = spectra.groupby(['run', 'generation'], sort=False).mean()
    run_generational_fitness = species.groupby(['run', 'generation'], sort=False).mean()

    fig, axes = plt.subplots(*shape, sharex=True, sharey='row', constrained_layout=True,
                             gridspec_kw={'height_ratios': [1.5, 1]}, **kwargs)
    runs = run_generational_spectrum.index.get_level_values('run').unique()

    for i, (row, col) in enumerate(itertools.product(range(0, axes.shape[0], 2), range(axes.shape[1]))):
        logger.debug('Plotting run number {}'.format(runs[i]))
        if base_run is None:
            run_spectrum = run_generational_spectrum.loc[(runs[i], slice(None)), :].sort_index(level='generation')
        else:
            run_spectrum = run_generational_spectrum.loc[((base_run, runs[i]), slice(None)), :].sort_index(
                level='generation')
        p = axes[row, col].pcolormesh(run_spectrum.T, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, col].tick_params(
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
            axes[row, col].set_title(runs[i])

        if base_run is None:
            data = run_generational_fitness.loc[(runs[i], slice(None)), :].sort_index(level='generation')
        else:
            data = run_generational_fitness.loc[((base_run, runs[i]), slice(None)), :].sort_index(
                level='generation')

        sns.lineplot(data=data.reset_index(), x='generation', y='fitness', label='mean', ax=axes[row + 1, col],
                     legend=False)
        sns.lineplot(data=data.reset_index(), x='generation', y='max', label='max', ax=axes[row + 1, col], legend=False,
                     alpha=0.5)
        if col == 0:
            axes[row + 1, col].set_ylabel('fitness')

    axes.flat[-1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.colorbar(p, ax=axes[0, -1], location='right')

    fig.suptitle(title)

    _plt_finish(view, filename, close)


def plot_run_spectra(spectra, cmap='rainbow', view=False, close=True, vmin=None, vmax=None, shape=(5, 5),
                     filename='spectra.svg', title="Communication Spectra for All Runs", base_run=None,
                     numbering=False, colorbar=True, **kwargs):
    # if not _plt_check(): return

    logger.debug('Compiling data')
    run_generational_spectrum = spectra.groupby(['run', 'generation'], sort=False).mean()

    fig, axes = plt.subplots(*shape, sharex=True, sharey=True, constrained_layout=True, **kwargs)
    axes = axes.reshape(*shape)
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

    print(filename)
    if colorbar:
        fig.colorbar(p, ax=axes[:, -1], location='right')

    if title is not None:
        fig.suptitle(title)

    _plt_finish(view, filename, close)


def plot_channel_volume_histogram(spectra: pd.DataFrame, view=False, close=True, vmin=None, vmax=None,
                                  filename='channel_histogram.svg', title="Channel volumes over multiple runs",
                                  fig=None, axes=None, **kwargs):
    if fig is None:
        fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, constrained_layout=True)
        fig.suptitle(title)
    else:
        assert axes is not None

    bins = np.arange(0, 0.7, step=0.1)

    for i, ax in enumerate(axes.flat):
        vh = spectra.iloc[:, i]
        ax.hist(vh, bins=bins, **kwargs)
        ax.set_title(i)

    _plt_finish(view, filename, close)

    return fig, axes


def plot_subspecies_abundances(summary, run=None, species=None, role=None, view=False, close=True,
                               filename='speciation.svg',
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

    counts = summary.loc[:, 'counts'].unstack('subspecies').sort_index(level='generation')
    if run is None:
        counts = counts.droplevel('run')

    fig, ax = plt.subplots()
    counts.plot.area(legend=False, ax=ax)

    ax.set_title("Speciation")
    ax.set_ylabel("Size per Species")
    ax.set_xlabel("Generations")

    plt.tight_layout()

    _plt_finish(view, filename, close)


# def plot_subspecies_counts(summary, run=None, species=None, role=None, view=False, close=True,
#                            filename='speciation.svg'):
#     """Visualizes speciation throughout evolution.
#
#     :param pd.DataFrame summary: The dataframe produced by subspecies_averages_and_counts.
#     :param view:
#     :param filename:
#     :return:
#     """
#
#     if plt is None:
#         logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
#         return
#
#     counts = summary['counts'].unstack('subspecies').count(axis=1).rename('counts')
#
#     sns.relplot(x='generation', y='counts', hue='role', kind='line', data=counts.reset_index())
#
#     ax = plt.gca()
#     ax.set_title("Fitness by Subspecies")
#     ax.set_ylabel("Subspecies fitness weighted by size")
#     ax.set_xlabel("Generations")
#
#     plt.tight_layout()
#
#     _plt_finish(view, filename, close)


def plot_pairplot(summary, run=None, species=None, role=None, view=False, close=True, filename='speciation.svg'):
    """Visualizes speciation throughout evolution.

    :param pd.DataFrame summary: The dataframe produced by subspecies_averages_and_counts.
    :param view:
    :param filename:
    :return:
    """

    if plt is None:
        logger.warning("This display is not available due to a missing optional dependency (matplotlib)")
        return

    sns.pairplot(data=summary.xs(role, level='role'), plot_kws={'alpha': 0.05})

    _plt_finish(view, filename, close)


def plot_subspecies_fitness(summary, run=None, species=None, role=None, view=False, close=True,
                            filename='subspecies_fitness.svg'):
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

    _plt_finish(view, filename, close)


def plot_subspecies_average_fitness(individuals: pd.DataFrame, run=None, species=None, role=None, view=False,
                                    close=True,
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

    _plt_finish(view, filename, close)


def plot_species_fitness(fitness: pd.DataFrame, run=None, species=None, role=None, only_mean=False, view=False,
                         close=True,
                         filename='fitness.svg', **kwargs):
    """Visualizes fitness levels for species

    :param pd.DataFrame fitness: The dataframe produced by subspecies_averages_and_counts.
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
        fitness = fitness.xs(xs_keys, level=xs_levels)

    if not only_mean:
        sns.lineplot(x="generation", y='max', data=fitness.reset_index(), **kwargs)
    sns.lineplot(x="generation", y='fitness', data=fitness.reset_index(), **kwargs)
    if not only_mean:
        sns.lineplot(x="generation", y='min', data=fitness.reset_index(), **kwargs)

    ax = plt.gca()
    ax.set_title("Fitness")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generations")

    plt.tight_layout()

    _plt_finish(view, filename, close)


def plot_species_fitnesses_by_run(fitness: pd.DataFrame, run=None, species=None, role=None, view=False, close=True,
                                  filename='fitness.svg', **kwargs):
    """Visualizes fitness levels for species

    :param pd.DataFrame fitness: The dataframe produced by subspecies_averages_and_counts.
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
        fitness = fitness.xs(xs_keys, level=xs_levels)

    fitness = fitness.join(fitness.groupby(['run']).mean()['fitness'].rename('run fitness'), on='run')

    sns.lineplot(x="generation", y='fitness', data=fitness.reset_index(), hue='run fitness', estimator=None, alpha=0.25,
                 **kwargs)

    ax = plt.gca()
    ax.set_title("Fitness")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generations")

    plt.tight_layout()

    _plt_finish(view, filename, close)


def plot_network_stats(data: pd.DataFrame, run=None, species=None, role=None, view=False, close=True,
                       axes=None,
                       title_1='nodes', title_2='connections', xlabel='generation', ylabel='count',
                       filename='fitness.svg', **kwargs):
    """Visualizes the average node and connection count for each generation

    :param pd.DataFrame fitness: The dataframe produced by subspecies_averages_and_counts.
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
        data = data.xs(xs_keys, level=xs_levels)

    ax = plt.gca()
    if axes is not None:
        ax = axes[0]
    # ax.xaxis.set_major_locator(FixedLocator(300))
    sns.lineplot(x="generation", y='nodes', data=data.reset_index(), ax=ax, **kwargs)
    ax.set_title('nodes')
    ax.set_ylabel('count')

    if title_1 is not None:
        ax.set_title(title_1)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if axes is not None:
        ax = axes[1]
        # ax.xaxis.set_major_locator(FixedLocator(300))
    sns.lineplot(x="generation", y='connections', data=data.reset_index(), ax=ax, **kwargs)
    if title_2 is not None:
        ax.set_title(title_2)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # ax = plt.gca()
    # ax.set_ylabel("Count")
    # ax.set_xlabel("Generations")
    # plt.gcf().suptitle("Neural Network Nodes and Connections")

    plt.tight_layout()

    _plt_finish(view, filename, close)


def plot_subspecies_count(data: pd.DataFrame, run=None, species=None, role=None, view=False, close=True,
                          filename='fitness.svg',
                          title="Number of Subspecies", xlabel='generation', ylabel='count',
                          **kwargs):
    """Visualizes fitness levels for species

    :param pd.DataFrame fitness: The dataframe produced by subspecies_averages_and_counts.
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
        data = data.xs(xs_keys, level=xs_levels)

    sns.lineplot(x="generation", y='subspecies', data=data.reset_index(), **kwargs)

    ax = plt.gca()
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    plt.tight_layout()

    _plt_finish(view, filename, close)
