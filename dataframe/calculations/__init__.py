from typing import Iterable

import pandas as pd


def individual(individuals: pd.DataFrame = None, individuals_file: str = None, save='individuals.parquet'):
    assert not (individuals is None and individuals_file is None)

    if individuals_file is not None:
        individuals = pd.read_pickle(individuals_file)

    individuals = individuals.drop(columns='genome')
    individuals = individuals.set_index(keys=['run', 'generation', 'species', 'subspecies', 'role'])

    if save is not None:
        individuals.to_parquet(save)

    return individuals


def species_fitness_summary(individuals: pd.DataFrame = None, individuals_file: str = None, save='fitness.parquet'):
    assert not (individuals is None and individuals_file is None)

    if individuals_file is not None:
        individuals = pd.read_pickle(individuals_file)

    fitness = individuals.groupby(['run', 'generation', 'species', 'role'])['fitness']
    fitness_summary = fitness.agg(['max', 'mean', 'min', 'std'])

    if save is not None:
        fitness_summary.to_parquet(save)

    return fitness_summary


def subspecies_averages_and_counts(individuals: pd.DataFrame = None, individuals_file: str = None,
                                   save='subspecies.parquet'):
    assert not (individuals is None and individuals_file is None)

    if individuals_file is not None:
        individuals = pd.read_pickle(individuals_file)

    bysubspecies = individuals.groupby(['run', 'generation', 'species', 'subspecies', 'role'])
    counts = bysubspecies['id'].count().rename('counts')
    subspecies_summary = bysubspecies[['fitness', 'nodes', 'connections']].mean().join(counts).dropna()

    if save is not None:
        subspecies_summary.to_parquet(save)

    return subspecies_summary


def species_averages(individuals: pd.DataFrame = None, individuals_file: str = None, save='species.parquet'):
    assert not (individuals is None and individuals_file is None)

    if individuals_file is not None:
        individuals = pd.read_pickle(individuals_file)

    byspecies = individuals.groupby(['run', 'generation', 'species', 'role'])
    species_summary = byspecies[['fitness', 'nodes', 'connections']].mean()

    if save is not None:
        species_summary.to_parquet(save)

    return species_summary


def specific_generation_over_runs(generations: Iterable[int], data: pd.DataFrame = None, data_file: str = None):
    """Select specific generations from a dataset.

    :param generations:
    :param data:
    :param data_file:
    :return:
    """
    assert not (data is None and data_file is None)

    if data_file is not None:
        data = pd.read_parquet(data_file)

    idx = pd.IndexSlice
    specific_generations = data.loc[idx[:, generations], :]

    return specific_generations
