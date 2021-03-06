from __future__ import print_function

import configparser
import logging
import os
import sys
from datetime import datetime
from multiprocessing.pool import Pool
from queue import Empty
from string import ascii_uppercase
from threading import Thread

import joblib
import neat
import numpy as np
import pandas as pd

EN_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
print(EN_PATH)
sys.path.append(EN_PATH)

from ..genome import DefaultGenome
from ..parallel import MultiQueue
from ..species import Species
from ..stats import DataFrameReporter
from ..noise import Noise, GenerationStepNoise
from ..dataframe import shrink_archive, shrink_individuals

config = configparser.ConfigParser()
config_file = os.path.abspath(os.path.join(__file__, '..', 'config.ini'))
print('Reading file at {}'.format(config_file))
config.read(config_file)

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
logging.basicConfig(level=logging.DEBUG, filename='{}/{}.log'.format(EN_PATH, now),
                    format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
print('Logging to {}'.format(os.path.abspath('{}/{}.log'.format(EN_PATH, now))))

rootlogger = logging.getLogger()
f = logging.Filter(name='evolvingniches')
logger = logging.getLogger('evolvingniches.run')
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(fmt='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'))
rootlogger.addHandler(sh)

# key = config['logdna']['key']
# print('Logging with key {}'.format(key))
# options = {'level': 'debug'}
# logdna_handler = LogDNAHandler(key, options)
# rootlogger.addHandler(logdna_handler)

rootlogger.addFilter(f)


def load_run_from_directory(directory):
    pass


def run(conf_encoders, conf_decoders, generations, view, noise_channel, noise_level, noise_generation, directory='',
        run_id=None, resume: str = None):
    """

    :param conf_encoders:
    :param conf_decoders:
    :param generations:
    :param view:
    :param noise_channel:
    :param noise_level:
    :param noise_generation:
    :param directory:
    :param run_id:
    :param resume: Path to the directory containing the checkpoint files to resume from.
    :return:
    """
    logger.info('Starting run id {}'.format(run_id))

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

    logger.debug('Dumping configs to {}'.format(os.path.abspath('data/{}/'.format(dirname))))

    joblib.dump(config_enc, 'data/{}/enc_config.pickle'.format(dirname))
    joblib.dump(config_dec, 'data/{}/dec_config.pickle'.format(dirname))
    joblib.dump(eval_config, 'data/{}/eval_config.pickle'.format(dirname))

    messages = MultiQueue()

    if resume is None:
        species = [
            Species.create(config_enc, config_dec, messages, run=run_id, checkpoint_dir='data/{}'.format(dirname),
                           evaluator_config=eval_config) for _ in range(1)]
        start_generation = 0
    else:
        logger.info('Resuming run from checkpoint file: {}'.format(resume))
        species = [Species.load(resume, messages, run=run_id, checkpoint_dir='data/{}'.format(dirname),
                                evaluator_config=eval_config)]
        start_generation = species[0].generation
        assert np.all([start_generation == s.generation for s in species])
        logger.debug('Starting at generation {}'.format(start_generation))

    # Set noise parameters
    setup_noise(noise_channel, noise_generation, noise_level, species, start_generation=start_generation)

    # Start statistics modules
    stats_mods = []

    dataframe_list = do_evolution(generations, species, stats_mods, start_generation)

    ind_file, mess_file = do_pandas(dataframe_list, dirname, species, start_generation)

    return mess_file, ind_file


def do_pandas(dataframe_list, dirname, species, start_generation):
    # DataFrame Data Storage #
    logging.debug('Creating DataFrame in {} for species {}...'.format(dirname, species))
    columns = [
        # 'id',
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
    # df.set_index(['run', 'id'])
    df = shrink_archive(df)
    df['generation'] = df['generation'] + start_generation
    logging.debug('Saving message DataFrame in {} for species {}...'.format(dirname, species))
    mess_file = 'data/{}/messages.parquet'.format(dirname)
    df.to_parquet(mess_file)
    individual_df = DataFrameReporter.dataframe()
    for s in species:
        individual_df = individual_df.append(s.encoder.df_reporter.df)
        individual_df = individual_df.append(s.decoder.df_reporter.df)
    individual_df = shrink_individuals(individual_df)
    logging.debug('Saving individual DataFrame in {} for species {}...'.format(dirname, species))
    ind_file = 'data/{}/individuals.xz'.format(dirname)
    # individual_df['generation'] = individual_df['generation'] + start_generation # I think this is extraneous
    individual_df.to_pickle(ind_file)
    return ind_file, mess_file


def setup_noise(noise_channel, noise_generation, noise_level, species, start_generation=0):
    if noise_generation is not None:
        n = GenerationStepNoise(noise_level, noise_channel, noise_generation, current_generation=start_generation)
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


def do_evolution(generations, species, stats_mods, start_generation=0):
    threads = [Thread(target=s.run) for s in stats_mods]
    for s in threads:
        s.start()
    # Run for up to 300 generations.
    k = start_generation
    dataframe_list = []
    while generations < 0 or k < generations:
        logger.info('Starting Generation {}'.format(k))
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
            logger.debug('Run Arguments: {}'.format(run_args))
            if args.resume is None:
                res = p.starmap(run, [run_args + [i + args.run_id] for i in range(args.runs)])
            else:
                res = p.starmap(run, [run_args + [i + args.run_id] + [args.resume] for i in range(args.runs)])
            p.close()
            p.join()
            if args.summary:
                for r in res:
                    messages_arch.append(pd.read_parquet(r[0]))
                    individuals.append(pd.read_pickle(r[1]))
    else:
        for r in range(args.runs):
            # ml, fl, sl,\
            mess_f, ind_f = run(args.encoder_conf, args.decoder_conf, args.generations, args.show,
                                args.noise_channel, args.noise_level, args.noise_generation,
                                args.dir, run_id=r + args.run_id, resume=args.resume)
            if args.summary:
                messages_arch.append(pd.read_parquet(mess_f))
                individuals.append(pd.read_pickle(ind_f))

    logger.debug('Path for summary info: {}'.format(os.path.abspath('')))
    dirname = setup_directories(args.dir, run_id=-1)

    joblib.dump(args, 'data/{}/args.pickle'.format(dirname))

    if args.summary:
        messages_df = pd.concat(messages_arch)
        messages_df.to_parquet('data/{}/messages.parquet'.format(dirname))
        individual_df = pd.concat(individuals)
        individual_df.to_pickle('data/{}/individuals.xz'.format(dirname))


def _get_parser():
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
    parser.add_argument('--run-id', metavar='R', type=int, action='store',
                        default=0,
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
    parser.add_argument('--summary', default=False, action='store_true', help='Compute a summary statistics file')

    return parser


if __name__ == '__main__':
    parser = _get_parser()

    logger.debug('Parsing Args.')
    args = parser.parse_args()
    logger.debug(args)
    main(args)
