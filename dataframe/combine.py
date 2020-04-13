import logging
import os, sys

import pandas as pd

EN_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
print(EN_PATH)
sys.path.append(EN_PATH)

logger = logging.getLogger('evolvingniches.dataframe.combine')

def combine_archives(folder):
    messages_files = []
    individuals_files = []

    for path, dirs, files in os.walk(folder):
        if 'messages.parquet' in files:
            messages_files.append(os.path.join(path, 'messages.parquet'))

        if 'individuals.xz' in files:
            individuals_files.append(os.path.join(path, 'individuals.xz'))

    messages = pd.read_parquet(messages_files[0])
    for i, f in enumerate(messages_files[1:]):
        print('Adding dataframe #{}: {}'.format(i + 2, f))
        m = pd.read_parquet(f)
        messages = messages.append(m)

    messages.to_parquet(os.path.join(folder, 'messages.parquet'))

    individuals = pd.read_pickle(individuals_files[0])
    for i, f in enumerate(individuals_files[1:]):
        print('Adding dataframe #{}: {}'.format(i + 2, f))
        i = pd.read_pickle(f)
        individuals = individuals.append(i)

    individuals.to_pickle(os.path.join(folder, 'individuals.xz'))

def main(args):
    combine_archives(args.dir)

if __name__ == '__main__':
    import argparse

    local_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='Combine the results of multiple runs.')
    parser.add_argument('dir', type=str, help='directory containing the files')

    logger.debug('Parsing Args.')
    args = parser.parse_args()
    logger.debug(args)
    main(args)
