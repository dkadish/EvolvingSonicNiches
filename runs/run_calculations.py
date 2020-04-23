import logging
import os

from dataframe.calculations import subspecies_averages_and_counts
from dataframe.calculations.spectrum import encoded_by_run_generation, received_by_run_generation

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
f = logging.Filter(name='evolvingniches')
logger = logging.getLogger('evolvingniches.run.calculations')
logger.addFilter(f)

def main(arguments):
    directory = arguments.dir

    encoded = encoded_by_run_generation(message_file=os.path.join(directory, 'messages.parquet'),
                                        save=os.path.join(directory,'encoded_spectrum.parquet'))

    received = received_by_run_generation(message_file=os.path.join(directory, 'messages.parquet'),
                                          save=os.path.join(directory,'received_spectrum.parquet'))

    subspecies_averages_and_counts(individuals_file=os.path.join(directory, 'individuals.xz'),
                                   save=os.path.join(directory,'subspecies.parquet'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate sub-dataframes for a run.')
    parser.add_argument('dir', type=str, default='', help='directory containing messages.parquet and individuals.xz')

    logger.debug('Parsing Args.')
    args = parser.parse_args()
    logger.debug(args)
    main(args)
