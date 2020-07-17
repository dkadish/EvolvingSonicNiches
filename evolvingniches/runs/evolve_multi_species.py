"""
"""

from __future__ import print_function

import os
from datetime import datetime
from itertools import chain
from string import ascii_uppercase
from subprocess import CalledProcessError
from threading import Thread

import joblib
import numpy as np

from evolvingniches.stats import Spectrum, Cohesion, Loudness, MessageSpectrum, Cluster, Messages

np.set_printoptions(precision=3)

import neat
from evolvingniches import visualize
from evolvingniches.parallel import MultiQueue
from evolvingniches.species import Species

inputs = [[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 1, 0],
          [1, 0, 1],
          [1, 1, 1],
          [0, 1, 1]]

N_MESSAGES = 10  # Number of messages to test on each individual in each evolutionary cycle

VIEW = False

N = 300
N_RUNS = 5


def plot_scores(now, dirname, species_id, species):
    # Visualize the scores
    decoding_scores = species.decoding_scores.get()
    species_gen_avg = []
    bits_gen_avg = []
    total_gen_avg = []
    species_gen_std = []
    bits_gen_std = []
    total_gen_std = []
    while decoding_scores is not False:
        # The decoding scores for an individual over an entire generation
        dc_species = decoding_scores['species']
        dc_bit = decoding_scores['bit']
        dc_total = decoding_scores['total']
        species_avg = []
        bits_avg = []
        total_avg = []

        for g in dc_species:
            species_avg.append(np.average(dc_species[g]))
            bits_avg.append(np.average(dc_bit[g]))
            total_avg.append(np.average(dc_total[g]))

        species_gen_avg.append(np.nanmean(species_avg))
        species_gen_std.append(np.nanstd(species_avg))
        bits_gen_avg.append(np.nanmean(bits_avg))
        bits_gen_std.append(np.nanstd(bits_avg))
        total_gen_avg.append(np.nanmean(total_avg))
        total_gen_std.append(np.nanstd(total_avg))

        decoding_scores = species.decoding_scores.get()

    visualize.plot_scores(np.array(species_gen_avg), np.array(species_gen_std),
                          np.array(bits_gen_avg), np.array(bits_gen_std),
                          np.array(total_gen_avg), np.array(total_gen_std),
                          view=VIEW,
                          filename='data/%s/%s-%i-scores.svg' % (
                              dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))

    return {
        'species': np.array(species_gen_avg),
        'bits': np.array(bits_gen_avg),
        'total': np.array(total_gen_avg)
    }


def plot_cohesion(cohesion_stats, now, dirname, species_id, loudness_stats, species):
    # Visualize the cohesion
    cohesion = [np.array(cohesion_stats.avg[species.species_id]), np.array(cohesion_stats.std[species.species_id])]
    loudness = [np.array(loudness_stats.avg[species.species_id]), np.array(loudness_stats.std[species.species_id])]
    visualize.plot_cohesion(cohesion[0], cohesion[1], loudness[0], loudness[1], view=VIEW,
                            filename='data/%s/%s-%i-message_cohesion.svg' % (
                                dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))


def plot_message_spectrum(now, dirname, species_id, message_spectrum_stats, species, spectrum_stats, vmin):
    # Visualize the spectra
    max_spectrum = max([np.max(np.array(spectrum_stats.spectra[s])) for s in spectrum_stats.spectra])
    spectra = spectrum_stats.spectra[species.species_id]
    visualize.plot_spectrum(spectra, view=VIEW, vmin=vmin, vmax=max_spectrum,
                            filename='data/%s/%s-%i-spectrum.svg' % (
                                dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    message_spectra = message_spectrum_stats.spectra[species.species_id]
    message_spectra['total'] = np.average([message_spectra[message] for message in message_spectra], axis=0)
    visualize.plot_message_spectrum(message_spectra, view=VIEW,
                                    filename='data/%s/%s-%i-message_spectrum.svg' % (
                                        dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    return message_spectra


def print_best(config_dec, config_enc, species_id, species):
    print('Stats for Species %i' % (species_id + 1))
    # Display the winning genome.
    print('\nBest {} genome:\n{!s}'.format(species_id, species.encoder.population.best_genome))
    print('\nBest {} genome:\n{!s}'.format(species_id, species.decoder.population.best_genome))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net_enc = neat.nn.FeedForwardNetwork.create(species.encoder.population.best_genome, config_enc)
    winner_net_dec = neat.nn.FeedForwardNetwork.create(species.decoder.population.best_genome, config_dec)
    for input in inputs[:3]:
        output_enc = winner_net_enc.activate(input)
        output_dec = winner_net_dec.activate(output_enc)
        print("Input {!r} -> {!r} -> {!r} Output".format(input, np.array(output_enc), np.array(output_dec)))
    node_names_enc = dict(zip(chain(range(-1, -4, -1), range(0, 9)), chain(range(0, 3), range(0, 9))))
    node_names_dec = dict(zip(chain(range(-1, -10, -1), range(0, 4)), chain(range(0, 9), range(0, 3), ['S'])))
    for n in node_names_dec: node_names_dec[n] = 'E{}'.format(node_names_dec[n]) if n < 0 else str(
        node_names_dec[n])
    for n in node_names_enc: node_names_enc[n] = str(node_names_enc[n]) if n < 0 else 'E{}'.format(
        node_names_enc[n])
    return node_names_dec, node_names_enc


def plot_networks(config_dec, config_enc, now, dirname, species_id, node_names_dec, node_names_enc, species):
    try:
        visualize.draw_net(config_dec, species.decoder.population.best_genome, view=False, prune_unused=True,
                           show_disabled=False, filename='data/%s/%s-%i-digraph_dec_pruned.gv' % (dirname,
                                                                                                  now.strftime(
                                                                                                      '%y-%m-%d_%H-%M-%S'),
                                                                                                  species_id),
                           node_names=node_names_dec)
        visualize.draw_net(config_enc, species.encoder.population.best_genome, view=False, prune_unused=True,
                           show_disabled=False, filename='data/%s/%s-%i-digraph_enc_pruned.gv' % (dirname,
                                                                                                  now.strftime(
                                                                                                      '%y-%m-%d_%H-%M-%S'),
                                                                                                  species_id),
                           node_names=node_names_enc)
    except CalledProcessError as e:
        print(e)


def plot_stats(now, dirname, species_id, species):
    visualize.plot_stats(species.encoder.stats, view=VIEW,
                         filename='data/%s/%s-%i-avg_fitness_enc.svg' % (
                             dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    visualize.plot_species(species.encoder.stats, view=VIEW,
                           filename='data/%s/%s-%i-speciation_enc.svg' % (
                               dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    visualize.plot_stats(species.decoder.stats, view=VIEW,
                         filename='data/%s/%s-%i-avg_fitness_dec.svg' % (
                             dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    visualize.plot_species(species.decoder.stats, view=VIEW,
                           filename='data/%s/%s-%i-speciation_dec.svg' % (
                               dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))


def run(config_encoders, config_decoders):
    # Load configuration.
    config_enc = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_encoders)
    config_dec = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_decoders)

    # encoded = MultiQueue()
    messages = MultiQueue()
    species = [Species(config_enc, config_dec,  # encoded,
                       messages, pairwise=False) for s in range(2)]

    # Start statistics modules
    spectrum_stats = Spectrum(messages.add())
    message_spectrum_stats = MessageSpectrum(messages.add())
    cohesion_stats = Cohesion(messages.add())
    loudness_stats = Loudness(messages.add())
    cluster_stats = Cluster(messages.add())
    messages_archive = Messages(messages.add())
    stats_mods = [spectrum_stats, message_spectrum_stats, cohesion_stats, loudness_stats, cluster_stats,
                  messages_archive]

    threads = [Thread(target=s.run) for s in stats_mods]
    for s in threads:
        s.start()

    # Run for up to 300 generations.
    n = N
    k = 0
    while n is None or k < n:

        k += 1

        for s in species:
            s.start()

        for s in species:
            s.join()

    for s in species:
        s.dataframe_list.put(False)

    for s, t in zip(stats_mods, threads):
        s.done()
        t.join()

    ####################################################################################################################

    vmin = 0

    dirname = '{0:%y}{1}{0:%d_%H%M}'.format(datetime.now(), ascii_uppercase[int(datetime.now().month) - 1])
    os.mkdir('data/{}'.format(dirname))

    pickle_file = 'data/{}/data.joblib'.format(dirname)
    pickle_data = {
        'config': {
            'sender': config_enc,
            'receiver': config_dec,
            'n_generations': n,
        },
        'message_spectra': {},
        'scores': {},
    }

    for i, s in enumerate(species):
        node_names_dec, node_names_enc = print_best(config_dec, config_enc, i, s)

        d = datetime.now()

        plot_networks(config_dec, config_enc, d, dirname, i, node_names_dec, node_names_enc, s)

        plot_stats(d, dirname, i, s)

        message_spectra = plot_message_spectrum(d, dirname, i, message_spectrum_stats, s, spectrum_stats, vmin)

        pickle_data['message_spectra'][s.species_id] = message_spectra

        plot_cohesion(cohesion_stats, d, dirname, i, loudness_stats, s)

        pickle_data['scores'][i] = plot_scores(d, dirname, i, s)

    ch = cluster_stats.ch
    silhouette = cluster_stats.silhouette
    archive = cluster_stats.archive

    pickle_data['ch'] = ch
    pickle_data['silhouette'] = silhouette
    pickle_data['archive'] = archive

    visualize.plot_clustering(ch, silhouette, archive, view=VIEW,
                              filename='data/%s/%s-clustering_stats' % (dirname, d.strftime('%y-%m-%d_%H-%M-%S')))

    print('Adding messages to pickle...')

    pickle_data['messages'] = {
        'original': messages_archive.originals,
        'encoded': messages_archive.encoded
    }

    print('Dumping data...')

    joblib.dump(pickle_data, pickle_file)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_encoders = os.path.join(local_dir, 'config-encoders')
    config_decoders = os.path.join(local_dir, 'config-decoders')

    for _ in range(N_RUNS):
        run(config_encoders, config_decoders)
