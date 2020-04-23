import pandas as pd


def subspecies_averages_and_counts(individuals: pd.DataFrame = None, individuals_file: str = None, save='subspecies.parquet'):
    assert not (individuals is None and individuals_file is None)

    if individuals_file is not None:
        individuals = pd.read_pickle(individuals_file)

    bysubspecies = individuals.groupby(['run', 'generation', 'species', 'subspecies', 'role'])
    counts = bysubspecies['id'].count().rename('counts')
    subspecies_summary = bysubspecies[['fitness', 'nodes', 'connections']].mean().join(counts).dropna()

    if save is not None:
        subspecies_summary.to_parquet(save)

    return subspecies_summary
