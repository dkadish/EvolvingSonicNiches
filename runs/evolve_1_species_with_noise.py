from __future__ import print_function

import logging
import os
import sys
from datetime import datetime
from multiprocessing.pool import Pool
from queue import Empty
from string import ascii_uppercase
from threading import Thread

import numpy as np
import pandas as pd

import neat

EN_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
print(EN_PATH)
sys.path.append(EN_PATH)

from genome import DefaultGenome
from parallel import MultiQueue
from species import Species
from stats import DataFrameReporter
from noise import Noise, GenerationStepNoise
from dataframe_archive import shrink_archive

np.set_printoptions(precision=3)

inputs = [[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 1, 0],
          [1, 0, 1],
          [1, 1, 1],
          [0, 1, 1]]

N_MESSAGES = 10  # Number of messages to test on each individual in each evolutionary cycle

N = 300
N_RUNS = 5

now = datetime.now()
logging.basicConfig(level=logging.DEBUG, filename='{}.log'.format(now), format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
print('Logging to {}'.format(os.path.abspath('{}.log'.format(now))))
f = logging.Filter(name='evolvingniches')
logger = logging.getLogger('evolvingniches.run')
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(fmt='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)
logger.addFilter(f)


def load_run_from_directory(directory):
    pass


def run(conf_encoders, conf_decoders, generations, view, noise_channel, noise_level, noise_generation, directory='',
        run_id=None, resume=None):
    logger.debug('Starting run id {}'.format(run_id))

    dirname = setup_directories(directory, run_id)

    # Load configuration
    config_enc = neat.Config(DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             conf_encoders)
    config_dec = neat.Config(DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             conf_decoders)

    eval_config = {
        'Simulation': {
            'n_messages': 7,
            'noise_overwrites_signal': False,
        },
        'Evaluation': {
            'correct_factor': 0.1,
            'loudness_penalty': 0.2,
            'no_species_id_score': True,
        }
    }

    messages = MultiQueue()
    species = [
        Species(config_enc, config_dec, messages, run=run_id, pairwise=False, checkpoint_dir='data/{}'.format(dirname),
                evaluator_config=eval_config) for _ in range(1)]

    # Set noise parameters
    setup_noise(noise_channel, noise_generation, noise_level, species)

    # Start statistics modules
    stats_mods = []

    dataframe_list = do_evolution(generations, species, stats_mods)

    ind_file, mess_file = do_pandas(dataframe_list, dirname, species)

    return mess_file, ind_file


def do_pandas(dataframe_list, dirname, species):
    # DataFrame Data Storage #
    logging.debug('Creating DataFrame in {} for species {}...'.format(dirname, species))
    columns = [
        'id',
        'run',
        'generation',
        'species',
        'sender',
        'receiver']
    columns.extend(['original_{}'.format(i) for i in range(3)])
    columns.extend(['encoded_{}'.format(i) for i in range(9)])
    columns.extend(['received_{}'.format(i) for i in range(9)])
    columns.extend([
        'score_identity',
        'score_bit',
        'score_total'
    ])
    df = pd.DataFrame(dataframe_list, columns=columns)
    df.set_index(['run', 'id'])
    df = shrink_archive(df)
    logging.debug('Saving message DataFramein {} for species {}...'.format(dirname, species))
    mess_file = 'data/{}/messages.parquet'.format(dirname)
    df.to_parquet(mess_file)
    individual_df = DataFrameReporter.dataframe()
    for s in species:
        individual_df = individual_df.append(s.encoder.df_reporter.df)
        individual_df = individual_df.append(s.decoder.df_reporter.df)
    logging.debug('Saving individual DataFrame in {} for species {}...'.format(dirname, species))
    ind_file = 'data/{}/individuals.xz'.format(dirname)
    individual_df.to_pickle(ind_file)
    return ind_file, mess_file


def setup_noise(noise_channel, noise_generation, noise_level, species):
    if noise_generation is not None:
        n = GenerationStepNoise(noise_level, noise_channel, noise_generation)
    else:
        n = Noise(noise_level, noise_channel)
    for s in species:
        s.encoder.evaluator.noise = n
        # s.encoder.evaluator.overwrite_message_with_noise()


def setup_directories(directory, run_id):
    """Set up the directory for the run data.

    Figure out what the directory should be called and then make it and return the path.

    :param directory: Base directory
    :param run_id: Which run (for multi-run simulations)
    :return: Path to the data directory
    """

    if run_id is None:
        logger.debug('No run_id')
        dirname = '{0:%y}{1}{0:%d_%H%M%S}'.format(datetime.now(), ascii_uppercase[int(datetime.now().month) - 1])
    else:
        logger.info('Running run number {}'.format(run_id + 1))
        run_id += 1
        dirname = '{0:%y}{1}{0:%d_%H%M%S}-{2}'.format(datetime.now(), ascii_uppercase[int(datetime.now().month) - 1],
                                                      run_id)
    if directory != '':
        os.makedirs('data/{}'.format(directory), exist_ok=True)
        dirname = '{}/{}'.format(directory, dirname)
    os.makedirs('data/{}'.format(dirname))
    return dirname


def do_evolution(generations, species, stats_mods):
    threads = [Thread(target=s.run) for s in stats_mods]
    for s in threads:
        s.start()
    # Run for up to 300 generations.
    k = 0
    dataframe_list = []
    while generations < 0 or k < generations:
        logger.debug('Starting Generation')
        k += 1
        for s in species:
            s.start()
        logger.debug('Waiting for species to finish.')
        while np.any([s.is_alive() for s in species]):
            for s in species:
                while True:
                    try:
                        row = s.dataframe_list.get(timeout=0.1)
                    except Empty:
                        break
                    dataframe_list.append(row)
                logger.debug('Trying to join')
                s.join(0.1)
                if not s.is_alive():
                    logger.debug('Finished species {}'.format(s.species_id))
    for s in species:
        s.dataframe_list.put(None)
    for s, t in zip(stats_mods, threads):
        s.done()
        t.join()

    return dataframe_list


def main(args):
    n = os.cpu_count() - 2
    run_args = [args.encoder_conf, args.decoder_conf, args.generations, args.show, args.noise_channel, args.noise_level,
                args.noise_generation, args.dir]
    messages_arch = []
    individuals = []
    if args.multiprocessing:
        with Pool(n) as p:
            logger.debug(run_args)
            res = p.starmap(run, [run_args + [i] for i in range(args.runs)])
            p.close()
            p.join()
            for r in res:
                messages_arch.append(pd.read_parquet(r[0]))
                individuals.append(pd.read_pickle(r[1]))
    else:
        for r in range(args.runs):
            # ml, fl, sl,\
            mess_f, ind_f = run(args.encoder_conf, args.decoder_conf, args.generations, args.show, args.noise_channel,
                                args.noise_level,
                                args.noise_generation, args.dir, run_id=r)
            messages_arch.append(pd.read_parquet(mess_f))
            individuals.append(pd.read_pickle(ind_f))

    logger.debug('Path for summary DataFrames: {}'.format(os.path.abspath('.')))
    dirname = setup_directories(args.dir, run_id=-1)

    messages_df = pd.concat(messages_arch)
    messages_df.to_parquet('data/{}/messages.parquet'.format(dirname))
    individual_df = pd.concat(individuals)
    individual_df.to_pickle('data/{}/individuals.xz'.format(dirname))


if __name__ == '__main__':
    import argparse

    local_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='Run evolution with a single species.')
    parser.add_argument('--encoder-conf', metavar='ENC', type=str, action='store',
                        default=os.path.join(local_dir, 'config-encoders'),
                        help='location of the encoder config file')
    parser.add_argument('--decoder-conf', metavar='DEC', type=str, action='store',
                        default=os.path.join(local_dir, 'config-decoders'),
                        help='location of the decoder config file')
    parser.add_argument('-g', '--generations', metavar='G', type=int, action='store',
                        default=N,
                        help='number of generations')
    parser.add_argument('-r', '--runs', metavar='R', type=int, action='store',
                        default=N_RUNS,
                        help='number of runs')
    parser.add_argument('-V', '--visualize', action='store_true', default=False,
                        help='Generate visualizations for the run.')
    parser.add_argument('-s', '--show', action='store_true', default=False,
                        help='Show visualizations for the run.')
    parser.add_argument('--noise-channel', type=int, default=0, nargs='*',
                        help='Which channel is the noise on?')
    parser.add_argument('--noise-level', type=float, default=0.0, nargs='*',
                        help='How much noise is there?')
    parser.add_argument('--noise-generation', type=int, default=None,
                        help='In which generation does the noise start?')
    parser.add_argument('-d', '--dir', type=str, default='', help='directory for the run')
    parser.add_argument('-m', '--multiprocessing', action='store_true', default=False,
                        help='use multiprocessing for the run')
    parser.add_argument('--resume', type=str, default=None, help='resume run from folder')

    logger.debug('Parsing Args.')
    args = parser.parse_args()
    logger.debug(args)
    main(args)
