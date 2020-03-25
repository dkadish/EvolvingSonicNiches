"""
"""

from __future__ import print_function

import os
import sys

from archive import Archive, MessageList

en_path = os.path.abspath(os.path.join(__file__, '..', '..'))
print(en_path)
sys.path.append(en_path)

from datetime import datetime
from multiprocessing.pool import Pool
from string import ascii_uppercase
from threading import Thread

import joblib
import numpy as np

import neat
from genome import DefaultGenome
from parallel import MultiQueue
from species import Species
from stats import Spectrum, Cohesion, Loudness, MessageSpectrum, Messages
from visualize.plot import plot_message_spectrum, plot_scores, plot_stats, plot_received_message_spectrum
from visualize.print import print_best
from noise import Noise, GenerationStepNoise

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


def run(conf_encoders, conf_decoders, generations, view, noise_channel, noise_level, noise_generation, directory='',
        run_id=None):
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
            'loudness_penalty': 0.1,
            'no_species_id_score': True,
        }
    }

    messages = MultiQueue()
    species = [Species(config_enc, config_dec, messages, pairwise=False, checkpoint_dir='data/{}'.format(dirname),
                       evaluator_config=eval_config) for _ in range(1)]

    # Set noise parameters
    if noise_generation is not None:
        n = GenerationStepNoise(noise_level, noise_channel, noise_generation)
    else:
        n = Noise(noise_level, noise_channel)
    for s in species:
        s.encoder.evaluator.noise = n
        # s.encoder.evaluator.overwrite_message_with_noise()

    # Start statistics modules
    spectrum_stats = Spectrum(messages.add())
    message_spectrum_stats = MessageSpectrum(messages.add())
    cohesion_stats = Cohesion(messages.add())
    loudness_stats = Loudness(messages.add())
    # cluster_stats = Cluster(messages.add())
    messages_archive = Messages(messages.add())
    stats_mods = [spectrum_stats, message_spectrum_stats, cohesion_stats, loudness_stats,  # cluster_stats,
                  messages_archive]

    threads = [Thread(target=s.run) for s in stats_mods]
    for s in threads:
        s.start()

    # Run for up to 300 generations.
    n = generations
    k = 0
    while n < 0 or k < n:
        k += 1
        for s in species:
            s.start()
        for s in species:
            s.join()

    for s in species:
        s.decoding_scores.put(False)

    for s, t in zip(stats_mods, threads):
        s.done()
        t.join()

    ####################################################################################################################

    vmin = 0

    pickle_file = 'data/{}/data.joblib'.format(dirname)
    pickle_data = {
        'config': {
            'sender': config_enc,
            'receiver': config_dec,
            'n_generations': n,
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

    d = datetime.now()
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

        pickle_data['scores'][i] = plot_scores(d, dirname, i, s)

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

    # NEW DATA STORAGE #
    arc_file = 'data/{}/archive.jbl'.format(dirname)
    a = Archive()
    ml = MessageList.from_message_archive(messages_archive)
    a.add_run(ml)
    a.save(arc_file)

    return ml

def main(args):
    n = os.cpu_count() - 2
    run_args = [args.encoder_conf, args.decoder_conf, args.generations, args.show, args.noise_channel, args.noise_level,
                args.noise_generation, args.dir]
    a = Archive()
    if args.multiprocessing:
        with Pool(n) as p:
            print(run_args)
            res = p.starmap(run, [run_args + [i] for i in range(args.runs)])
            p.close()
            p.join()
            for r in res:
                a.add_run(r)
    else:
        message_archives = []
        for r in range(args.runs):
            ma = run(args.encoder_conf, args.decoder_conf, args.generations, args.show, args.noise_channel, args.noise_level,
                args.noise_generation, args.dir, run_id=r)
            a.add_run(ma)

    a.save(os.path.join(args.dir, 'archive.jbl'))

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

    args = parser.parse_args()
    print(args)
    main(args)
