from typing import Iterable

import pandas as pd

from dataframe import Columns


def encoded_by_run_generation(messages: pd.DataFrame = None, message_file: str = None, save='encoded.parquet'):
    return _by_run_generation(Columns.encoded, messages, message_file, save)


def received_by_run_generation(messages: pd.DataFrame = None, message_file: str = None, save='received.parquet'):
    return _by_run_generation(Columns.received, messages, message_file, save)


def _by_run_generation(message_cols: list, messages: pd.DataFrame = None, message_file: str = None, save=None):
    """Produces a MultiIndex-ed dataframe with the indices being (run, generation, species).

    You must pass a list of the columns to load (encoded or received) and ONE OF the message DataFrame or the parquet
    file containing the message dataframe.

    :param message_cols: Columns to load
    :param messages: The dataframe to filter (optional)
    :param message_file: The file to load the dataframe from (optional)
    :return: pd.DataFrame
    """
    assert not (messages is None and message_file is None)

    if message_file is not None:
        cols = ['run', 'generation', 'species'] + message_cols
        messages = pd.read_parquet(message_file, columns=cols)

    run_generation_mean = messages.groupby(by=['run', 'generation', 'species']).mean()

    if save is not None:
        run_generation_mean.to_parquet(save)

    return run_generation_mean

def specific_generation_over_runs(generations: Iterable[int], spectra: pd.DataFrame = None, spectra_file: str = None):
    assert not (spectra is None and spectra_file is None)

    if spectra_file is not None:
        spectra = pd.read_parquet(spectra_file)

    idx = pd.IndexSlice
    specific_generations = spectra.loc[idx[:, generations], :]

    return specific_generations