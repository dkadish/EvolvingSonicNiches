import pandas as pd


def subspecies_averages_and_counts(individuals: pd.DataFrame, save=None):
    # subspecies_counts_by_generation = pd.pivot_table(individuals, values='id', index=['run','generation','species','role'], columns='subspecies', aggfunc=np.size, fill_value=0)
    # return subspecies_counts_by_generation

    bysubspecies = individuals.groupby(['run', 'generation', 'species', 'subspecies', 'role'])
    counts = bysubspecies['id'].count().rename('counts')
    subspecies_summary = bysubspecies[['fitness', 'nodes', 'connections']].mean().join(counts).dropna()

    if save is not None:
        subspecies_summary.to_parquet(save)

    return subspecies_summary