import logging
import os
from typing import Iterable

import pandas as pd

from dataframe.calculations import subspecies_averages_and_counts, species_averages, individual, species_fitness_summary
from dataframe.calculations.spectrum import encoded_by_run_generation, received_by_run_generation

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
f = logging.Filter(name='evolvingniches')
logger = logging.getLogger('evolvingniches.run.calculations')
logger.addFilter(f)

def exists(d, *args):
    for f in args:
        if not os.path.exists(os.path.join(d, f)):
            return False

    return True

def main(arguments):
    directory = arguments.dir

    if not exists(directory, 'encoded_spectrum.parquet', 'received_spectrum.parquet') or arguments.force:
        logger.info('Loading messages')
        messages = pd.read_parquet(os.path.join(directory, 'messages.parquet'))

        if not exists(directory, 'encoded_spectrum.parquet') or arguments.force:
            logger.info('Extracting encoded spectrum.')
            encoded = encoded_by_run_generation(messages=messages,
                                                save=os.path.join(directory, 'encoded_spectrum.parquet'))

        if not exists(directory, 'received_spectrum.parquet') or arguments.force:
            logger.info('Extracting received spectrum.')
            received = received_by_run_generation(messages=messages,
                                                  save=os.path.join(directory, 'received_spectrum.parquet'))

    if not exists(directory, 'subspecies.parquet', 'fitness.parquet', 'species.parquet', 'individuals.parquet') \
            or arguments.force:
        logger.info('Loading individuals')

        try:
            individuals = pd.read_parquet(os.path.join(directory, 'individuals.parquet'))
        except OSError as e:
            individuals = pd.read_pickle(os.path.join(directory, 'individuals.xz'))

        if not exists(directory, 'individuals.parquet') or arguments.force:
            individual(individuals=individuals, save=os.path.join(directory, 'individuals.parquet'))

        if not exists(directory, 'fitness.parquet') or arguments.force:
            logger.info('Extracting species fitness summary.')
            species_fitness_summary(individuals=individuals, save=os.path.join(directory, 'fitness.parquet'))

        if not exists(directory, 'subspecies.parquet') or arguments.force:
            logger.info('Extracting subspecies averages and counts.')
            subspecies_averages_and_counts(individuals=individuals,
                                           save=os.path.join(directory, 'subspecies.parquet'))

        if not exists(directory, 'species.parquet') or arguments.force:
            logger.info('Extracting species averages.')
            species_averages(individuals=individuals,
                             save=os.path.join(directory, 'species.parquet'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate sub-dataframes for a run.')
    parser.add_argument('dir', type=str, default='', help='directory containing messages.parquet and individuals.xz')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='force regeneration of files')

    logger.debug('Parsing Args.')
    a = parser.parse_args()
    logger.debug(a)
    main(a)
