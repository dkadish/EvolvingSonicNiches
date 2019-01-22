"""
"""

from __future__ import print_function
import os
import random
from datetime import datetime
from multiprocessing import Manager, Queue, Process
from threading import Thread

import neat
import visualize
from evaluators import EncoderEvaluator, DecoderEvaluator

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

    # Create the population, which is the top-level object for a NEAT run.
    enc = neat.Population(config_enc)
    dec = neat.Population(config_dec)

    # Add a stdout reporter to show progress in the terminal.
    enc.add_reporter(neat.StdOutReporter(True))
    enc_stats = neat.StatisticsReporter()
    enc.add_reporter(enc_stats)
    enc.add_reporter(neat.Checkpointer(5))

    dec.add_reporter(neat.StdOutReporter(True))
    dec_stats = neat.StatisticsReporter()
    dec.add_reporter(dec_stats)
    dec.add_reporter(neat.Checkpointer(5))

    encoded = Queue()
    scores = Queue()
    encoder_genomes = Queue()
    decoder_genomes = Queue()
    enc_evaluator = EncoderEvaluator(encoded, scores, encoder_genomes)
    dec_evaluator = DecoderEvaluator(encoded, scores, decoder_genomes)

    # Run for up to 300 generations.
    n = 10
    k = 0
    while n is None or k < n:

        k += 1

        p_enc = Thread(target=enc.run_generation, args=(enc_evaluator.evaluate,))
        p_dec = Thread(target=dec.run_generation, args=(dec_evaluator.evaluate,))

        print('Started Encoder')
        p_enc.start()
        print('Started Decoder')
        p_dec.start()

        p_enc.join()
        p_dec.join()

        if enc.config.no_fitness_termination:
            enc.reporters.found_solution(enc.config, enc.generation, enc.best_genome)

        if dec.config.no_fitness_termination:
            dec.reporters.found_solution(enc.config, dec.generation, dec.best_genome)

    ####################################################################################################################

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(enc.best_genome))
    print('\nBest genome:\n{!s}'.format(dec.best_genome))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net_enc = neat.nn.FeedForwardNetwork.create(enc.best_genome, config_enc)
    winner_net_dec = neat.nn.FeedForwardNetwork.create(dec.best_genome, config_dec)
    for input in inputs:
        output_enc = winner_net_enc.activate(input)
        output_dec = winner_net_dec.activate(output_enc)
        # print("input {!r}, expected output {!r}, got {!r}".format(input, output_enc))
        # print("input {!r}, expected output {!r}, got {!r}".format(input, output_dec))

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    d = datetime.now()
    visualize.draw_net(config_enc, enc.best_genome, True, filename='%s-digraph_enc.gv' %d.strftime('%y-%m-%d_%H-%M-%S'))#, node_names=node_names)
    visualize.draw_net(config_dec, dec.best_genome, True, filename='%s-digraph_dec.gv' %d.strftime('%y-%m-%d_%H-%M-%S'))#, node_names=node_names)
    visualize.plot_stats(enc_stats, ylog=False, view=True, filename='%s-avg_fitness_enc.svg' %d.strftime('%y-%m-%d_%H-%M-%S'))
    # visualize.plot_species(enc_stats, view=True)
    visualize.plot_stats(dec_stats, ylog=False, view=True, filename='%s-avg_fitness_dec.svg' %d.strftime('%y-%m-%d_%H-%M-%S'))
    # visualize.plot_species(dec_stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_encoders = os.path.join(local_dir, 'config-encoders')
    config_decoders = os.path.join(local_dir, 'config-decoders')
    run(config_encoders, config_decoders)