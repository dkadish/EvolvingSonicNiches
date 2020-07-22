"""
"""

from __future__ import print_function

import os
import random
from datetime import datetime
from multiprocessing import Manager, Queue, Process
from threading import Thread

import neat

from evolvingniches import visualize

inputs = [[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 1, 0],
          [1, 0, 1],
          [1, 1, 1],
          [0, 1, 1]]

N_MESSAGES = 10  # Number of messages to test on each individual in each evolutionary cycle


# TODO Can the evaluators query the generation of the evaluation to stay synced? Or the number of individuals? This would avoid the need for the edit to Population
class Evaluation:

    def __init__(self):
        self.manager = Manager()
        self.messages = self.manager.list()
        self.encoded_messages = Queue()
        self.encoder_scores = Queue()
        self.encoder_genomes = Queue()
        self.decoder_genomes = Queue()

    def eval_encoders(self, genomes, config):
        # print("Evaluating Encoders")
        self.messages.extend(
            [[[int(random.random() > 0.5) for i in range(3)] for j in range(N_MESSAGES)] for k in range(len(genomes))])
        encoded_messages = []

        # Create the NNs and encode the messages
        for i, (genome_id, genome) in enumerate(genomes):
            # print("Evaluating Encoder number %i" %i)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            messages = []
            for m in self.messages[i]:
                messages.append(net.activate(m))
            encoded_messages.append(messages)

        self.encoded_messages.put(encoded_messages)

        # Wait for the decoded messages from the decoders and evaluate
        encoder_scores = self.encoder_scores.get()
        for score, (genome_id, genome) in zip(encoder_scores, genomes):
            genome.fitness = score

        self.encoder_genomes.put(genomes)

    def eval_decoders(self, genomes, config):
        # print("Evaluating Decoders")
        # Wait for encoded messages from encoders
        encoded_messages = self.encoded_messages.get()

        # Set up scoring structures
        encoder_scores = [0 for i in range(len(encoded_messages))]

        # Create the NNs and encode the messages
        for genome_id, genome in genomes:
            score = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for i, ind_messages in enumerate(encoded_messages):
                for j, message in enumerate(ind_messages):
                    decode = net.activate(message)

                    diff = sum([abs(o - d) for o, d in zip(self.messages[i][j], decode)])

                    encoder_scores[i] -= diff
                    score -= diff

            genome.fitness = score

        self.encoder_scores.put(encoder_scores)

        self.decoder_genomes.put(genomes)


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

    # Run for up to 300 generations.
    n = 50
    k = 0
    while n is None or k < n:

        evaluation = Evaluation()
        k += 1

        def eval_enc(genomes, c):
            p = Process(target=evaluation.eval_encoders, args=(genomes, c))
            p.start()
            g = evaluation.encoder_genomes.get()
            for fit_gene, gene in zip(g, genomes):
                gene[1].fitness = fit_gene[1].fitness
            p.join()

        def eval_dec(genomes, c):
            p = Process(target=evaluation.eval_decoders, args=(genomes, c))
            p.start()
            g = evaluation.decoder_genomes.get()
            for fit_gene, gene in zip(g, genomes):
                gene[1].fitness = fit_gene[1].fitness
            p.join()

        enc_status = Queue()
        dec_status = Queue()

        def run_enc_gen():
            solution = enc.run_generation(eval_enc)
            enc_status.put(solution)

        def run_dec_gen():
            solution = dec.run_generation(eval_dec)
            dec_status.put(solution)

        p_enc = Thread(target=run_enc_gen)
        p_dec = Thread(target=run_dec_gen)

        p_enc.start()
        p_dec.start()

        enc_solution = enc_status.get()
        dec_solution = dec_status.get()

        p_enc.join()
        p_dec.join()

        if not enc_solution and not dec_solution:
            break

        if enc.config.no_fitness_termination:
            enc.reporters.found_solution(enc.config, enc.generation, enc.best_genome)

        if dec.config.no_fitness_termination:
            dec.reporters.found_solution(enc.config, dec.generation, dec.best_genome)

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
    visualize.draw_net(config_enc, enc.best_genome, True,
                       filename='%s-digraph_enc.gv' % d.strftime('%y-%m-%d_%H-%M-%S'))  # , node_names=node_names)
    visualize.draw_net(config_dec, dec.best_genome, True,
                       filename='%s-digraph_dec.gv' % d.strftime('%y-%m-%d_%H-%M-%S'))  # , node_names=node_names)
    visualize.plot_stats(enc_stats, ylog=False, view=True,
                         filename='%s-avg_fitness_enc.svg' % d.strftime('%y-%m-%d_%H-%M-%S'))
    # visualize.plot_species(enc_stats, view=True)
    visualize.plot_stats(dec_stats, ylog=False, view=True,
                         filename='%s-avg_fitness_dec.svg' % d.strftime('%y-%m-%d_%H-%M-%S'))
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
