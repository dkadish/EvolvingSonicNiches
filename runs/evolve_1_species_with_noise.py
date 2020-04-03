from __future__ import print_function

import logging
import os
import sys

from datetime import datetime
from multiprocessing.pool import Pool
from queue import Empty
from string import ascii_uppercase
from threading import Thread
import joblib
import numpy as np
import pandas as pd
import neat

EN_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
print(EN_PATH)
sys.path.append(EN_PATH)

from genome import DefaultGenome
from parallel import MultiQueue
from species import Species
from stats import Spectrum, Cohesion, Loudness, MessageSpectrum, Messages, DataFrameReporter
from visualize.plot import plot_message_spectrum, plot_stats, plot_received_message_spectrum#, \
    # get_decoding_scores_list, plot_scores
from visualize.print import print_best
from noise import Noise, GenerationStepNoise
from archive import Archive
from archive.messages import MessageList
from archive.fitness import FitnessList
from archive.score import ScoreList

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

logging.basicConfig(level=logging.DEBUG)
f = logging.Filter(name='evolvingniches')
logger = logging.getLogger('evolvingniches.run')
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
    species = [Species(config_enc, config_dec, messages, run=run_id, pairwise=False, checkpoint_dir='data/{}'.format(dirname),
                       evaluator_config=eval_config) for _ in range(1)]

    # Set noise parameters
    setup_noise(noise_channel, noise_generation, noise_level, species)

    # Start statistics modules
    message_spectrum_stats, messages_archive, spectrum_stats, stats_mods = setup_stats(messages)

    dataframe_list = do_evolution(generations, species, stats_mods)

    ####################################################################################################################

    vmin = 0

    pickle_file = 'data/{}/data.joblib'.format(dirname)
    # TODO This should become a data archive class
    pickle_data = {
        'config': {
            'sender': config_enc,
            'receiver': config_dec,
            'n_generations': generations,
        },
        'evaluator_config': eval_config,
        'encoder_stats': {},
        'decoder_stats': {},
        'message_spectra': {},
        'received_message_spectra': {},
        'scores': {},
        'generations': generations,
        'noise_channel': noise_channel,
        'noise_level': noise_level
    }

    ####### Plot and Pickle #########
    species_scores = {}
    for i, s in enumerate(species):
        node_names_dec, node_names_enc = print_best(config_dec, config_enc, i, s)

        d = datetime.now()

        # plot_networks(config_dec, config_enc, d, dirname, i, node_names_dec, node_names_enc, s, view=view)

        plot_stats(d, dirname, i, s, view=True)

        # TODO: Something is wrong with this module.
        message_spectra = plot_message_spectrum(d, dirname, i, message_spectrum_stats, s, spectrum_stats, vmin,
                                                view=view)
        received_message_spectra = plot_received_message_spectrum(d, dirname, i, message_spectrum_stats, s,
                                                                  spectrum_stats, vmin,
                                                                  view=view)

        pickle_data['message_spectra'][s.species_id] = message_spectra
        pickle_data['received_message_spectra'][s.species_id] = received_message_spectra

        # plot_cohesion(cohesion_stats, d, dirname, i, loudness_stats, s, view=view)

        # scores = get_decoding_scores_list(s)
        # species_scores[s.species_id] = scores
        # pickle_data['scores'][i] = plot_scores(d, dirname, i, scores)

        pickle_data['encoder_stats'][s.species_id] = s.encoder.stats
        pickle_data['decoder_stats'][s.species_id] = s.decoder.stats

    print('Adding messages to pickle...')

    pickle_data['messages'] = {
        'original': messages_archive.originals,
        'encoded': messages_archive.encoded,
        'received': messages_archive.received,
    }

    print('Dumping data...')

    joblib.dump(pickle_data, pickle_file)

    # DataFrame Data Storage #
    print('Creating DataFrame...')
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

    # df_list = []
    # while True:
    #     row = species.dataframe_list.get()
    #     if row is None:
    #         break
    #     df_list.append(row)

    df = pd.DataFrame(dataframe_list, columns=columns)
    df.set_index(['run','id'])
    print('Saving DataFrame...')
    mess_file = 'data/{}/dataframe_archive.xz'.format(dirname)
    df.to_pickle(mess_file)

    individual_df = DataFrameReporter.dataframe()
    for s in species:
        individual_df = individual_df.append(s.encoder.df_reporter.df)
        individual_df = individual_df.append(s.decoder.df_reporter.df)
    print('Saving DataFrame...')
    ind_file = 'data/{}/dataframe_individuals.xz'.format(dirname)
    individual_df.to_pickle(ind_file)

    # NEW DATA STORAGE #
    print('Creating Python Class Archive...')
    arc_file = 'data/{}/archive.jbl'.format(dirname)
    a = Archive()
    ml = MessageList.from_message_archive(messages_archive, run_id)

    fl = FitnessList()
    sl = ScoreList()
    for s in species:
        # sl.extend(ScoreList.from_score_list(species_scores[s.species_id], run=run_id, species=s.species_id))
        fl.extend(FitnessList.from_statistics_reporters(s.encoder.stats, s.decoder.stats, run=run_id, species=s.species_id))

    a.add_run(ml, fl, sl)
    print('Saving archive...')
    a.save(arc_file)

    return ml, fl, sl, mess_file, ind_file

def setup_stats(messages):
    spectrum_stats = Spectrum(messages.add())
    message_spectrum_stats = MessageSpectrum(messages.add())
    cohesion_stats = Cohesion(messages.add())
    loudness_stats = Loudness(messages.add())
    # cluster_stats = Cluster(messages.add())
    messages_archive = Messages(messages.add())
    stats_mods = [spectrum_stats, message_spectrum_stats, cohesion_stats, loudness_stats,  # cluster_stats,
                  messages_archive]
    return message_spectrum_stats, messages_archive, spectrum_stats, stats_mods


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
        print('No run_id')
        dirname = '{0:%y}{1}{0:%d_%H%M%S}'.format(datetime.now(), ascii_uppercase[int(datetime.now().month) - 1])
    else:
        print('Running run number {}'.format(run_id + 1))
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
    a = Archive()
    messages_arch = []
    individuals = []
    if args.multiprocessing:
        with Pool(n) as p:
            print(run_args)
            res = p.starmap(run, [run_args + [i] for i in range(args.runs)])
            p.close()
            p.join()
            for r in res:
                a.add_run(r[0], r[1], r[2])
                messages_arch.append(pd.read_pickle(r[3]))
                individuals.append(pd.read_pickle(r[4]))
    else:
        message_archives = []
        for r in range(args.runs):
            ml, fl, sl, mess_f, ind_f = run(args.encoder_conf, args.decoder_conf, args.generations, args.show, args.noise_channel, args.noise_level,
                args.noise_generation, args.dir, run_id=r)
            a.add_run(ml, fl, sl)
            messages_arch.append(pd.read_pickle(mess_f))
            individuals.append(pd.read_pickle(ind_f))

    print(os.path.abspath('.'))
    dirname = setup_directories(args.dir, run_id=-1)

    messages_df = pd.concat(messages_arch)
    messages_df.to_pickle('data/{}/messages.xz'.format(dirname))
    individual_df = pd.concat(individuals)
    individual_df.to_pickle('data/{}/individuals.xz'.format(dirname))

    d = 'data/{}/archive.jbl.xz'.format(dirname)
    a.save(d)
    logger.info('Saving log file to {}'.format(d))
    print(os.path.abspath(d))

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
    print(args)
    main(args)
