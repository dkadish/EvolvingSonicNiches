import argparse
import os
import shutil
import types

import pandas as pd

from dataframe import shrink_individuals
from runs import run_calculations

directory = '/Users/davk/Documents/phd/projects/EvolvingNiches/data'
# directory = '/home/davk/EvolvingNiches/runs/data'

directories = []
individual_files = []
individual_file = 'individuals.parquet'
print('Finding Files...')
for d in os.listdir(directory):
    try:
        if individual_file in os.listdir(os.path.join(directory, d)):
            print(os.path.join(directory, d, individual_file))
            directories.append(os.path.join(directory, d))
            individual_files.append(os.path.join(directory, d, individual_file))
    except NotADirectoryError as e:
        pass

print()

# for f in individual_files:
#     print('Processing {}'.format(f))
#     print('Creating Backup...')
#     shutil.copyfile(f, f + '.backup')
#     print('Reading File...')
#     individuals = pd.read_parquet(f)
#     print('Adjusting generations...')
#     individuals = individuals.reset_index()
#     # individuals.loc[individuals['generation'] >= 600, 'generation'] -= 300
#     individuals = shrink_individuals(individuals)
#     individuals = individuals.set_index(keys=['run', 'generation', 'species', 'subspecies', 'role'])
#     print('Saving File...')
#     individuals.to_parquet(f)

for d in directories:

    print('Processing {}'.format(d))
    arguments = types.new_class('Args')()
    arguments.dir = d
    arguments.force = False
    arguments.config = None #os.path.join(directory,'..')

    print('Running calculations...')
    run_calculations.main(arguments)

