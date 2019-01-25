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
    firstSpecies = Species(config_enc, config_dec, encoded)
    secondSpecies = Species(config_enc, config_dec, encoded)

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

    ####################################################################################################################

    for i,s in enumerate([firstSpecies, secondSpecies]):
        # Display the winning genome.
        print('\nBest %i genome:\n{!s}'.format(i, s.encoder.population.best_genome))
        print('\nBest %i genome:\n{!s}'.format(i, s.decoder.population.best_genome))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net_enc = neat.nn.FeedForwardNetwork.create(s.encoder.population.best_genome, config_enc)
        winner_net_dec = neat.nn.FeedForwardNetwork.create(s.decoder.population.best_genome, config_dec)
        for input in inputs:
            output_enc = winner_net_enc.activate(input)
            output_dec = winner_net_dec.activate(output_enc)
            # print("input {!r}, expected output {!r}, got {!r}".format(input, output_enc))
            # print("input {!r}, expected output {!r}, got {!r}".format(input, output_dec))

        # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
        d = datetime.now()
        try:
            visualize.draw_net(config_enc, s.encoder.population.best_genome, view=False, filename='%s-%i-digraph_enc.gv' %(d.strftime('%y-%m-%d_%H-%M-%S'),i))#, node_names=node_names)
            visualize.draw_net(config_dec, s.decoder.population.best_genome, view=False, filename='%s-%i-digraph_dec.gv' %(d.strftime('%y-%m-%d_%H-%M-%S'),i))#, node_names=node_names)
        except CalledProcessError as e:
            print(e)

        visualize.plot_stats(s.encoder.stats, ylog=False, view=True, filename='%s-%i-avg_fitness_enc.svg' %(d.strftime('%y-%m-%d_%H-%M-%S'),i))
        # visualize.plot_species(enc_stats, view=True)
        visualize.plot_stats(s.decoder.stats, ylog=False, view=True, filename='%s-%i-avg_fitness_dec.svg' %(d.strftime('%y-%m-%d_%H-%M-%S'),i))
        # visualize.plot_species(dec_stats, view=True)

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
        cohesion = s.cohesion.get()
        while cohesion is not False:
            cohesions.append(np.average(list(cohesion.values())))

            cohesion = s.cohesion.get()

        visualize.plot_cohesion(cohesions, view=True,
                                filename='%s-%i-message_cohesion.svg' % (d.strftime('%y-%m-%d_%H-%M-%S'), i))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_encoders = os.path.join(local_dir, 'config-encoders')
    config_decoders = os.path.join(local_dir, 'config-decoders')
    run(config_encoders, config_decoders)