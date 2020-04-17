import numpy as np
import pandas as pd


def calculate_species_counts(individuals: pd.DataFrame):
    subspecies_counts_by_generation = pd.pivot_table(individuals, values='id', index=['run','generation','species','role'], columns='subspecies', aggfunc=np.size, fill_value=0)

    return subspecies_counts_by_generation