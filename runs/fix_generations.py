import argparse
import os, sys
import shutil
import types

import pandas as pd

EN_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
print(EN_PATH)
sys.path.append(EN_PATH)

from runs import run_calculations

directory = '/home/davk/EvolvingNiches/runs/data'

directories = []
individual_files = []
for d in os.listdir(directory):
    try:
        if 'individuals.xz' in os.listdir(os.path.join(directory, d)):
            print(os.path.join(directory, d, 'individuals.xz'))
            directories.append(os.path.join(directory, d))
            individual_files.append(os.path.join(directory, d, 'individuals.xz'))
    except NotADirectoryError as e:
        pass

for f in individual_files:
    shutil.copyfile(f, f + '.backup')
    individuals = pd.read_pickle(f)
    individuals.loc[individuals['generation'] >= 600, 'generation'] -= 300
    individuals.to_pickle(f)

for d in directories:

    arguments = types.new_class('Args')()
    arguments.dir = d
    arguments.force = True

    run_calculations.main(arguments)

