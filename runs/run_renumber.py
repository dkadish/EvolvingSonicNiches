import logging
import os

import pandas as pd

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
f = logging.Filter(name='evolvingniches')
logger = logging.getLogger('evolvingniches.run.renumber')
logger.addFilter(f)

def main(arguments):
    directory = arguments.dir
    offset = arguments.offset

    messages = pd.read_parquet(os.path.join(directory,'messages.parquet'))
    logger.info('Original run number: {}'.format(messages['run'][0]))
    messages['run'] += offset
    logger.info('Adjusted run number: {}'.format(messages['run'][0]))
    messages.to_parquet(os.path.join(directory,'messages.parquet'))


    individuals = pd.read_pickle(os.path.join(directory,'individuals.xz'))
    logger.info('Original run number: {}'.format(individuals['run'][0]))
    individuals['run'] += offset
    logger.info('Adjusted run number: {}'.format(individuals['run'][0]))
    individuals.to_pickle(os.path.join(directory,'individuals.xz'))

    logger.info('Renumbering complete.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Renumber the run for a particular run for use as a base run for later simulations.')
    parser.add_argument('dir', type=str, default='', help='directory containing messages.parquet and individuals.xz')
    parser.add_argument('--offset', type=int, default=100, help='number to offset the run by')

    logger.debug('Parsing Args.')
    args = parser.parse_args()
    logger.debug(args)
    main(args)
