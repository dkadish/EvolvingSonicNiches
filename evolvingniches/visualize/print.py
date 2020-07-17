from itertools import chain

import neat
import numpy as np

# TODO Refactor this out to somewhere
inputs = [[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 1, 0],
          [1, 0, 1],
          [1, 1, 1],
          [0, 1, 1]]


def print_best(config_dec, config_enc, species_id, species):
    print('Stats for Species %i' % (species_id + 1))
    # Display the winning genome.
    print('\nBest {} genome:\n{!s}'.format(species_id, species.encoder.population.best_genome))
    print('\nBest {} genome:\n{!s}'.format(species_id, species.decoder.population.best_genome))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net_enc = neat.nn.FeedForwardNetwork.create(species.encoder.population.best_genome, config_enc)
    winner_net_dec = neat.nn.FeedForwardNetwork.create(species.decoder.population.best_genome, config_dec)
    for i in inputs[:3]:
        output_enc = winner_net_enc.activate(i)
        output_dec = winner_net_dec.activate(output_enc)
        print("Input {!r} -> {!r} -> {!r} Output".format(i, np.array(output_enc), np.array(output_dec)))
    node_names_enc = dict(zip(chain(range(-1, -4, -1), range(0, 9)), chain(range(0, 3), range(0, 9))))
    node_names_dec = dict(zip(chain(range(-1, -10, -1), range(0, 4)), chain(range(0, 9), range(0, 3), ['S'])))
    for n in node_names_dec:
        node_names_dec[n] = 'E{}'.format(node_names_dec[n]) if n < 0 else str(node_names_dec[n])
    for n in node_names_enc:
        node_names_enc[n] = str(node_names_enc[n]) if n < 0 else 'E{}'.format(node_names_enc[n])
    return node_names_dec, node_names_enc
