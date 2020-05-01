import os
from typing import Iterable

import numpy as np
import pandas as pd

import neat
from dataframe import shrink_individuals
from genome import DefaultGenome

def average_bit_scores(messages: pd.DataFrame = None, messages_file: str = None, save='scores.parquet'):
    assert not (messages is None and messages_file is None)

    if messages_file is not None:
        messages = pd.read_parquet(messages_file, columns=['run', 'generation', 'species', 'score_bit'])

    bit_scores = messages.groupby(['run', 'generation', 'species']).mean()

    if save is not None:
        bit_scores.to_parquet(save)

    return bit_scores

def diversity(species: pd.DataFrame, conf_encoders, conf_decoders, individuals: pd.DataFrame = None, individuals_file: str = None):
    assert not (individuals is None and individuals_file is None)

    if individuals_file is not None:
        individuals = pd.read_pickle(individuals_file)

    individuals = individuals.set_index(keys=['run', 'generation', 'species', 'subspecies', 'role', 'id'])

    config = neat.Config(DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         conf_encoders)

    species['diversity'] = 0.0

    for i in species.index:
        genomes = individuals.xs(i, level=species.index.names)['genome']

        distances = np.zeros(shape=(genomes.size, genomes.size))

        for g in genomes:
            for h in genomes[g.key:]:
                distances[g.key - 1, h.key - 1] = g.distance(h, config.genome_config)

        distance = np.average(distances)

        species.loc[i, 'diversity'] = distance

    return species


def individual(individuals: pd.DataFrame = None, individuals_file: str = None, save='individuals.parquet'):
    assert not (individuals is None and individuals_file is None)

    if individuals_file is not None:
        individuals = pd.read_pickle(individuals_file)

    individuals = individuals.drop(columns='genome')
    individuals = individuals.set_index(keys=['run', 'generation', 'species', 'subspecies', 'role'])

    if save is not None:
        individuals.to_parquet(save)

    return individuals


def species_summary(individuals: pd.DataFrame = None, individuals_file: str = None, save='species.parquet', config_path=None):
    assert not (individuals is None and individuals_file is None)

    if individuals_file is not None:
        individuals = pd.read_pickle(individuals_file)

    individuals = individuals.reset_index()
    individuals = shrink_individuals(individuals)

    # Column means
    no_subspecies_id = individuals.drop(columns=['subspecies', 'id'])
    byspecies = no_subspecies_id.groupby(['run', 'generation', 'species', 'role'])
    species_means = byspecies.mean().dropna()

    # Fitness summary stats
    fitness = individuals.groupby(['run', 'generation', 'species', 'role'])['fitness']
    fitness_summary = fitness.agg(['max', 'min', 'std']).dropna()

    summary = pd.concat([species_means, fitness_summary], axis=1)

    # Number of subspecies
    subspecies_count = individuals.reset_index().groupby(['run', 'generation', 'species', 'role'])[
        'subspecies'].nunique()
    summary = summary.join(subspecies_count, how='inner')

    if config_path is not None:
        enc = os.path.join(config_path, 'config-encoders')
        dec = os.path.join(config_path,'config-decoders')
        summary = diversity(summary, enc, dec, individuals=individuals)

    if save is not None:
        summary.to_parquet(save)

    return summary


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
