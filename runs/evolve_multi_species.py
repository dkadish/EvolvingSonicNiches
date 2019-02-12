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

from stats import Spectrum, Cohesion, Loudness, MessageSpectrum, Cluster

np.set_printoptions(precision=3)

import neat
import visualize
from parallel import MultiQueue
from species import Species

inputs = [[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 1, 0],
          [1, 0, 1],
          [1, 1, 1],
          [0, 1, 1]]

N_MESSAGES = 10  # Number of messages to test on each individual in each evolutionary cycle


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
    stats_mods = [spectrum_stats, message_spectrum_stats, cohesion_stats, loudness_stats, cluster_stats]

    threads = [Thread(target=s.run) for s in stats_mods]
    for s in threads:
        s.start()

    # Run for up to 300 generations.
    n = 300
    k = 0
    while n is None or k < n:

        k += 1

        for s in species:
            s.start()

        for s in species:
            s.join()

        # if enc.config.no_fitness_termination:
        #     enc.reporters.found_solution(enc.config, enc.generation, enc.best_genome)
        #
        # if dec.config.no_fitness_termination:
        #     dec.reporters.found_solution(enc.config, dec.generation, dec.best_genome)

    for s in species:
        s.decoding_scores.put(False)

    for s, t in zip(stats_mods, threads):
        s.done()
        t.join()

    ####################################################################################################################

    vmin = 0

    dirname = '{0:%y}{1}{0:%d_%H%M}'.format(datetime.now(), ascii_uppercase[int(datetime.now().month) - 1])
    os.mkdir('data/{}'.format(dirname))

    pickle_file = 'data/{}/data.joblib'.format(dirname)
    pickle_data = {}

    for i, s in enumerate(species):
        print('Stats for Species %i' % (i + 1))
        # Display the winning genome.
        print('\nBest {} genome:\n{!s}'.format(i, s.encoder.population.best_genome))
        print('\nBest {} genome:\n{!s}'.format(i, s.decoder.population.best_genome))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net_enc = neat.nn.FeedForwardNetwork.create(s.encoder.population.best_genome, config_enc)
        winner_net_dec = neat.nn.FeedForwardNetwork.create(s.decoder.population.best_genome, config_dec)
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

        d = datetime.now()
        try:
            visualize.draw_net(config_dec, s.decoder.population.best_genome, view=False, prune_unused=True,
                               show_disabled=False, filename='data/%s/%s-%i-digraph_dec_pruned.gv' % (dirname,
                                                                                                      d.strftime(
                                                                                                          '%y-%m-%d_%H-%M-%S'),
                                                                                                      i),
                               node_names=node_names_dec)
            visualize.draw_net(config_enc, s.encoder.population.best_genome, view=False, prune_unused=True,
                               show_disabled=False, filename='data/%s/%s-%i-digraph_enc_pruned.gv' % (dirname,
                                                                                                      d.strftime(
                                                                                                          '%y-%m-%d_%H-%M-%S'),
                                                                                                      i),
                               node_names=node_names_enc)
        except CalledProcessError as e:
            print(e)

        visualize.plot_stats(s.encoder.stats, view=True,
                             filename='data/%s/%s-%i-avg_fitness_enc.svg' % (
                             dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))
        visualize.plot_species(s.encoder.stats, view=True,
                               filename='data/%s/%s-%i-speciation_enc.svg' % (
                               dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))
        visualize.plot_stats(s.decoder.stats, view=True,
                             filename='data/%s/%s-%i-avg_fitness_dec.svg' % (
                             dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))
        visualize.plot_species(s.decoder.stats, view=True,
                               filename='data/%s/%s-%i-speciation_dec.svg' % (
                               dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))

        # Visualize the spectra
        max_spectrum = max([np.max(np.array(spectrum_stats.spectra[s])) for s in spectrum_stats.spectra])
        spectra = spectrum_stats.spectra[s.species_id]

        visualize.plot_spectrum(spectra, view=True, vmin=vmin, vmax=max_spectrum,
                                filename='data/%s/%s-%i-spectrum.svg' % (dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))

        message_spectra = message_spectrum_stats.spectra[s.species_id]
        message_spectra['total'] = np.average([message_spectra[message] for message in message_spectra], axis=0)

        visualize.plot_message_spectrum(message_spectra, view=True,
                                        filename='data/%s/%s-%i-message_spectrum.svg' % (
                                        dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))

        # Visualize the cohesion
        cohesion = [np.array(cohesion_stats.avg[s.species_id]), np.array(cohesion_stats.std[s.species_id])]
        loudness = [np.array(loudness_stats.avg[s.species_id]), np.array(loudness_stats.std[s.species_id])]

        visualize.plot_cohesion(cohesion[0], cohesion[1], loudness[0], loudness[1], view=True,
                                filename='data/%s/%s-%i-message_cohesion.svg' % (
                                dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))

        # Visualize the scores
        decoding_scores = s.decoding_scores.get()
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

            decoding_scores = s.decoding_scores.get()

        visualize.plot_scores(np.array(species_gen_avg), np.array(species_gen_std),
                              np.array(bits_gen_avg), np.array(bits_gen_std),
                              np.array(total_gen_avg), np.array(total_gen_std),
                              view=True,
                              filename='data/%s/%s-%i-scores.svg' % (dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))

    # Visualize the cluster stats
    # cohesion = [np.array(cohesion_stats.avg[s.species_id]), np.array(cohesion_stats.std[s.species_id])]
    # loudness = [np.array(loudness_stats.avg[s.species_id]), np.array(loudness_stats.std[s.species_id])]
    #
    # visualize.plot_cohesion(cohesion[0], cohesion[1], loudness[0], loudness[1], view=True,
    #                         filename='data/%s/%s-%i-message_cohesion.svg' % (dirname, d.strftime('%y-%m-%d_%H-%M-%S'), i))
    ch = cluster_stats.ch
    silhouette = cluster_stats.silhouette
    archive = cluster_stats.archive

    pickle_data['ch'] = ch
    pickle_data['silhouette'] = silhouette
    pickle_data['archive'] = archive

    visualize.plot_clustering(ch, silhouette, archive, view=True,
                              filename='data/%s/%s-clustering_stats' % (dirname, d.strftime('%y-%m-%d_%H-%M-%S')))

    joblib.dump(pickle_data, pickle_file)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_encoders = os.path.join(local_dir, 'config-encoders')
    config_decoders = os.path.join(local_dir, 'config-decoders')
    run(config_encoders, config_decoders)
