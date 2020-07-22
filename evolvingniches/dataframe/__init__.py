import gc
import os

import neat
import pandas as pd


class Columns:
    # id = 'id'
    run = 'run'
    generation = 'generation'
    species = 'species'
    sender = 'sender'
    receiver = 'receiver'
    original = ['original_{}'.format(i) for i in range(3)]
    encoded = ['encoded_{}'.format(i) for i in range(9)]
    received = ['received_{}'.format(i) for i in range(9)]
    score_identity = 'score_identity'
    score_bit = 'score_bit'
    score_total = 'score_total'

    # identifiers = [id, run, generation, species, sender, receiver]
    identifiers = [run, generation, species, sender, receiver]
    messages = original + encoded + received
    scores = [score_identity, score_bit, score_total]

    all = identifiers + messages + scores


def shrink_individuals(individuals):
    downcasts = {
        'unsigned': ['id', 'run', 'generation', 'species', 'subspecies', 'nodes', 'connections'],
        'float': ['fitness']
    }

    categories = ['role']

    for d in downcasts:
        for col in downcasts[d]:
            individuals.loc[:, col] = pd.to_numeric(individuals.loc[:, col], downcast=d)

    individuals.info()
    for col in categories:
        individuals.loc[:, col] = individuals.loc[:, col].astype('category')

    return individuals


def shrink_archive(archive):
    drop = ['id']
    downcasts = {
        'unsigned': ['run', 'generation', 'species', 'sender', 'receiver'],
        'float': ['encoded_{}'.format(i) for i in range(9)] + ['received_{}'.format(i) for i in range(9)] + [
            'score_bit', 'score_total']
    }
    boolean = ['original_{}'.format(i) for i in range(3)] + ['score_identity']
    sparse = ['encoded_{}'.format(i) for i in range(9)] + ['received_{}'.format(i) for i in range(9)]

    for d in drop:
        if d in archive.columns:
            archive = archive.drop(columns=d)

    for d in downcasts:
        for col in downcasts[d]:
            archive.loc[:, col] = pd.to_numeric(archive.loc[:, col], downcast=d)

    for col in boolean:
        archive.loc[:, col] = archive.loc[:, col].astype(bool)

    # if use_sparse:
    #     for col in sparse:
    #         archive.loc[:, col] = pd.arrays.SparseArray(archive.loc[:, col], fill_value=0.0)

    return archive


def combine_archives(folder):
    df_archives, df_inds = [], []
    for path, dirs, files in os.walk(folder):
        if 'dataframe.xz' in files:
            df_archives.append(os.path.join(path, 'dataframe.xz'))

        if 'dataframe_individuals.xz' in files:
            df_inds.append(os.path.join(path, 'dataframe_individuals.xz'))

    archives = shrink_archive(pd.read_pickle(df_archives[0]))
    for i, f in enumerate(df_archives[1:]):
        gc.collect()
        print('Adding message dataframe #{}: {}'.format(i + 2, f))
        a = shrink_archive(pd.read_pickle(f))
        archives = archives.append(a)
        print('{} dataframes, using {} MB'.format(i + 2, archives.memory_usage(index=True).sum() * pow(10, -6)))

        # archives.to_pickle(os.path.join(folder, 'messages.xz'))
        archives.to_parquet(os.path.join(folder, 'messages.parquet'))
        print('Finished pickling...')

    gc.collect()
    print('Finished appending. Pickling now.')
    # archives.to_pickle(os.path.join(folder, 'messages.xz'))
    archives.to_parquet(os.path.join(folder, 'messages.parquet'))

    individuals = pd.read_pickle(df_inds[0])
    for i, f in enumerate(df_inds[1:]):
        print('Adding individual dataframe #{}: {}'.format(i + 2, f))
        a = pd.read_pickle(f)
        individuals = individuals.append(a)
        print('{} dataframes, using {} MB'.format(i + 2, individuals.memory_usage(index=True).sum() * pow(10, -6)))

    individuals.to_pickle(os.path.join(folder, 'individuals.xz'))


class EnvironmentConfig:
    class Simulation:

        def __init__(self, n_messages=7, noise_overwrites_signal=False, n_generations=300,
                     noise_channel=[4], noise_level=[1.0]):
            self.n_messages = n_messages
            self.noise_overwrites_signal = noise_overwrites_signal
            self.n_generations = n_generations
            self.noise_channel = noise_channel
            self.noise_level = noise_level

    class Evaluation:

        def __init__(self, correct_factor=0.1, loudness_penalty=0.1, no_species_id_score=True):
            self.correct_factor = correct_factor
            self.loudness_penalty = loudness_penalty
            self.no_species_id_score = no_species_id_score


class Config:

    def __init__(self, sender: neat.config.Config, receiver: neat.config.Config, environment: EnvironmentConfig):
        self.sender = sender
        self.receiver = receiver
        self.environment = environment


class Message:

    @staticmethod
    def create():
        columns = [
            # 'id',
            'run',
            'generation',
            'species',
            'sender',
            'receiver']

        columns.extend(['original_{}'.format(i) for i in range(3)])
        columns.extend(['encoded_{}'.format(i) for i in range(9)])
        columns.extend(['received_{}'.format(i) for i in range(9)])

        columns.extend([
            'score_identity',
            'score_bit',
            'score_total'
        ])

        return pd.DataFrame(columns=columns)
