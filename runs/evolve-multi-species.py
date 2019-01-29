"""
"""

from __future__ import print_function
import os
import random
from datetime import datetime
from multiprocessing import Manager, Queue, Process
from subprocess import CalledProcessError
from threading import Thread

import numpy as np

import neat
import visualize
from evaluators import EncoderEvaluator, DecoderEvaluator
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

N_MESSAGES = 10 # Number of messages to test on each individual in each evolutionary cycle


def run(config_encoders, config_decoders):
    # Load configuration.
    config_enc = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_encoders)
    config_dec = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_decoders)

    encoded = MultiQueue()
    firstSpecies = Species(config_enc, config_dec, encoded, pairwise=True)
    secondSpecies = Species(config_enc, config_dec, encoded, pairwise=True)

    # Run for up to 300 generations.
    n = 150
    k = 0
    while n is None or k < n:

        k += 1

        firstSpecies.start()
        secondSpecies.start()

        firstSpecies.join()
        secondSpecies.join()

        # if enc.config.no_fitness_termination:
        #     enc.reporters.found_solution(enc.config, enc.generation, enc.best_genome)
        #
        # if dec.config.no_fitness_termination:
        #     dec.reporters.found_solution(enc.config, dec.generation, dec.best_genome)

    firstSpecies.spectra.put(False)
    secondSpecies.spectra.put(False)
    firstSpecies.cohesion.put(False)
    secondSpecies.cohesion.put(False)
    firstSpecies.decoding_scores.put(False)
    secondSpecies.decoding_scores.put(False)

    ####################################################################################################################

    for i,s in enumerate([firstSpecies, secondSpecies]):
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
            print("Input {!r} -> {!r} -> {!r} Output".format(input, output_enc, output_dec))

        # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
        d = datetime.now()
        try:
            visualize.draw_net(config_enc, s.encoder.population.best_genome, view=False,
                               filename='%s-%i-digraph_enc.gv' % (
                               d.strftime('%y-%m-%d_%H-%M-%S'), i))  # , node_names=node_names)
            visualize.draw_net(config_dec, s.decoder.population.best_genome, view=False,
                               filename='%s-%i-digraph_dec.gv' % (
                               d.strftime('%y-%m-%d_%H-%M-%S'), i))  # , node_names=node_names)
            visualize.draw_net(config_dec, s.decoder.population.best_genome, view=False, prune_unused=True,
                               show_disabled=False, filename='%s-%i-digraph_dec_pruned.gv' % (
                d.strftime('%y-%m-%d_%H-%M-%S'), i))  # , node_names=node_names)
            visualize.draw_net(config_enc, s.encoder.population.best_genome, view=False, prune_unused=True,
                               show_disabled=False, filename='%s-%i-digraph_enc_pruned.gv' % (
                d.strftime('%y-%m-%d_%H-%M-%S'), i))  # , node_names=node_names)
        except CalledProcessError as e:
            print(e)

        visualize.plot_stats(s.encoder.stats, view=True,
                             filename='%s-%i-avg_fitness_enc.svg' % (d.strftime('%y-%m-%d_%H-%M-%S'), i))
        visualize.plot_species(s.encoder.stats, view=True,
                               filename='%s-%i-speciation_enc.svg' % (d.strftime('%y-%m-%d_%H-%M-%S'), i))
        visualize.plot_stats(s.decoder.stats, view=True,
                             filename='%s-%i-avg_fitness_dec.svg' % (d.strftime('%y-%m-%d_%H-%M-%S'), i))
        visualize.plot_species(s.decoder.stats, view=True,
                               filename='%s-%i-speciation_dec.svg' % (d.strftime('%y-%m-%d_%H-%M-%S'), i))

        # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
        # p.run(eval_genomes, 10)

        # Visualize the spectra
        spectra = []
        spectrum = s.spectra.get()
        while spectrum is not False:
            spectra.append(spectrum)

            spectrum = s.spectra.get()

        visualize.plot_spectrum(spectra, view=True,
                             filename='%s-%i-spectrum.svg' % (d.strftime('%y-%m-%d_%H-%M-%S'), i))

        # Visualize the cohesion
        cohesions = []
        loudness_avg = []
        loudness_std = []
        cohesion_array = s.cohesion.get()
        while cohesion_array is not False:
            cohesion, loudness = cohesion_array
            cohesions.append(np.average(list(cohesion.values())))
            loudness_avg.append(loudness['avg'])
            loudness_std.append(loudness['std'])

            cohesion_array = s.cohesion.get()

        visualize.plot_cohesion(cohesions, np.array(loudness_avg), np.array(loudness_std), view=True,
                                filename='%s-%i-message_cohesion.svg' % (d.strftime('%y-%m-%d_%H-%M-%S'), i))

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
                              view=True, filename='%s-%i-scores.svg' % (d.strftime('%y-%m-%d_%H-%M-%S'), i))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_encoders = os.path.join(local_dir, 'config-encoders')
    config_decoders = os.path.join(local_dir, 'config-decoders')
    run(config_encoders, config_decoders)
