import numpy as np
import pandas as pd

from dataframe import Columns


def calculate_encoded_spectrum(messages: pd.DataFrame = None, message_file: str = None):
    return _calculate_spectrum(Columns.encoded, messages, message_file)

def calculate_received_spectrum(messages: pd.DataFrame = None, message_file: str = None):
    return _calculate_spectrum(Columns.received, messages, message_file)

def _calculate_spectrum(message_cols: list, messages: pd.DataFrame = None, message_file: str = None):
    assert not (messages is None and message_file is None)

    if message_file is not None:
        cols = ['run', 'generation', 'species'] + message_cols
        messages = pd.read_parquet(message_file, columns=cols)

    generation_mean = messages.groupby(by=['run', 'generation', 'species']).mean()

    spectrum = generation_mean.loc[:, message_cols]

    return spectrum.to_numpy()


def calculate_species_counts(individuals: pd.DataFrame):
    subspecies_counts_by_generation = pd.pivot_table(individuals, values='id', index=['run','generation','species','role'], columns='subspecies', aggfunc=np.size, fill_value=0)

    return subspecies_counts_by_generation