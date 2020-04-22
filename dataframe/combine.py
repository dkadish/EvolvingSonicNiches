import logging
import os, sys
from datetime import datetime

import pandas as pd

EN_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
print(EN_PATH)
sys.path.append(EN_PATH)

now = datetime.now()
logging.basicConfig(level=logging.DEBUG, filename='{}.log'.format(now), format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
print('Logging to {}'.format(os.path.abspath('{}.log'.format(now))))
f = logging.Filter(name='evolvingniches')
logger = logging.getLogger('evolvingniches.dataframe.combine')
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(fmt='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)
logger.addFilter(f)

def combine_archives(folder):
    messages_files = []
    individuals_files = []

    logger.debug('Appending Filepaths')
    for path, dirs, files in os.walk(folder):
        if 'messages.parquet' in files:
            messages_files.append(os.path.join(path, 'messages.parquet'))

        if 'individuals.xz' in files:
            individuals_files.append(os.path.join(path, 'individuals.xz'))

    logger.debug('Combining message archives')
    print('Adding dataframe #{}: {}'.format(1, messages_files[0]))
    messages = pd.read_parquet(messages_files[0])
    for i, f in enumerate(messages_files[1:]):
        print('Adding dataframe #{}: {}'.format(i + 2, f))
        m = pd.read_parquet(f)
        messages = messages.append(m)

    messages.to_parquet(os.path.join(folder, 'messages.parquet'))

    print('Adding dataframe #{}: {}'.format(1, individuals_files[0]))
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
