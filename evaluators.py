import random
from itertools import product
from math import tanh
from multiprocessing import Queue
from random import choice

import numpy as np
from scipy.spatial.distance import cdist

import neat
from messaging import Message, MessageType

N_MESSAGES = 10

MESSAGE_SET = [
          # tuple([0, 0, 0]),
          tuple([0, 0, 1]),
          tuple([0, 1, 0]),
          tuple([0, 1, 1]),
          tuple([1, 0, 0]),
          tuple([1, 0, 1]),
          tuple([1, 1, 0]),
          tuple([1, 1, 1])
]

CORRECT_FACTOR = 0.1

NULL = False

def nonlin_fitness(x):
    f = (tanh(8.0 * (x - 0.5)) + 1.0) / 2.0
    return f

def correct_multiplier(fitness, original, decode):
    multiplier = 1.0 + CORRECT_FACTOR
    for o, d in zip(original, decode):
        if abs(o - d) < 0.5:
            fitness *= multiplier
            multiplier += CORRECT_FACTOR

    return fitness

class BaseEvaluator:

    def __init__(self, messages: Queue, scores: Queue, genomes: Queue,
                 decoding_scores: Queue, species_id=0):
        self.messages = messages

        self.scores = scores
        self.genomes = genomes

        self.process = None

        self.decoding_scores = decoding_scores

        self.species_id = species_id  # For identifying members of the same species in multispecies processes.

    def __del__(self):
        pass

    def evaluate(self, genomes, config):
        self.evaluator(genomes, config)

        g = self.genomes.get()
        for genome_id, genome in genomes:
            genome.fitness = g[genome_id].fitness

    def evaluator(self, genomes, config):
        # Reset and zero all fitnesses
        for id, g in genomes:
            g.fitness = 0


class EncoderEvaluator(BaseEvaluator):

    def __init__(self, messages: Queue, scores: Queue, genomes: Queue,
                 decoding_scores: Queue, species_id=0):
        super(EncoderEvaluator, self).__init__(messages, scores, genomes,
                 decoding_scores, species_id)

        self._randomized = False

    def set_randomized_messages(self):
        self._randomized = True

    @property
    def randomized_messages(self):
        return self._randomized

    def evaluator(self, genomes, config):
        super(EncoderEvaluator, self).evaluator(genomes, config)

        # Generate a set of messages to use for testing
        if self._randomized:
            # print('Using randomized messages.')
            messages = [[tuple([int(random.random() > 0.5) for i in range(3)]) for j in range(N_MESSAGES)] for k in
                        range(len(genomes))]
        else:
            # print('Using standardized message set.')
            messages = [MESSAGE_SET for k in range(len(genomes))]

        # Create the NNs and encode the messages
        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for original_message in messages[i]:
                encoded_message = net.activate(original_message)
                # self.encoded.put((self.species_id, genome_id, original_message, encoded_message))
                self.messages.put(Message.Encoded(self.species_id, genome_id, original_message, encoded_message))

        # Send Finished Message
        # self.encoded.put(False) # FIXME encorporate into messages
        self.messages.put(Message.Generation(self.species_id))

        # Wait for the scores from the decoders and evaluate
        scores = self.scores.get()

        genome_dict = dict(genomes)
        for genome_id in genome_dict:
            try:
                genome_dict[genome_id].fitness = scores[genome_id]
            except KeyError as e:
                print('{0} not in fitness dictionary. Setting to 0.'.format(genome_id))

        self.genomes.put(genome_dict)


class DecoderEvaluator(BaseEvaluator):

    def evaluator(self, genomes, config):
        super(DecoderEvaluator, self).evaluator(genomes, config)

        encoder_fitness = {}
        species_scores = {}
        bit_scores = {}
        total_scores = {}
        species_ids = []
        false_count = 0

        # Wait for encoded messages from encoders
        message = self.messages.get()  # Assume the first one isn't false
        while message.type != MessageType.FINISHED:
            enc_species_id = message.species_id
            enc_genome_id = message.message['genome_id']
            original_message = message.message['original']
            encoded_message = message.message['encoded']

            # Update available species IDs, test for completeness
            if enc_species_id not in species_ids:
                species_ids.append(enc_species_id)

            if enc_genome_id not in encoder_fitness and enc_species_id == self.species_id:
                encoder_fitness[enc_genome_id] = 0

            for genome_id, genome in genomes:
                if genome_id not in species_scores:
                    species_scores[genome_id] = []
                    bit_scores[genome_id] = []
                    total_scores[genome_id] = []

                net = neat.nn.FeedForwardNetwork.create(genome, config)
                decoded_message = net.activate(encoded_message)

                # Ensure decoded_message is between 0 and 1
                decoded_message = np.clip(decoded_message, 0, 1)

                e_fitness = DecoderEvaluator.sender_fitness(original_message, decoded_message, enc_species_id, self.species_id, NULL)
                d_fitness = DecoderEvaluator.receiver_fitness(original_message, decoded_message, enc_species_id, self.species_id, NULL)
                species_score, bit_score, total_score = DecoderEvaluator.score(original_message, decoded_message, enc_species_id, self.species_id)

                if e_fitness is not None:
                    encoder_fitness[enc_genome_id] += e_fitness

                if d_fitness is not None:
                    genome.fitness += d_fitness

                species_scores[genome_id].append(species_score)
                if bit_score is not None or total_score is not None:
                    assert bit_score is not None and total_score is not None

                    bit_scores[genome_id].append(bit_score)
                    total_scores[genome_id].append(total_score)

            message = self.messages.get()

            do_break = False
            while message.type == MessageType.GENERATION:
                false_count += 1
                if false_count == len(species_ids):
                    print('Stopped with %i False values' % false_count)
                    print(species_ids)
                    do_break = True
                    break
                message = self.messages.get()
            if do_break:
                break

        self.scores.put(encoder_fitness)

        self.genomes.put(dict(genomes))

        self.decoding_scores.put({'species': species_scores, 'bit': bit_scores, 'total': total_scores})

    @staticmethod
    def sender_fitness(original, decode, sender_species, receiver_species, null=False):
        '''Score the sender.

        The sender only gets points for receivers within the species. Other receivers are ignored.
        Maximum theoretical fitness is 4 * n_messages * n_individuals = 4*7*50 = 1400.

        :param original:
        :param decode:
        :param sender_species:
        :param receiver_species:
        :return:
        '''
        decided_same = decode[-1] >= 0.5  # NN thinks its from the same species
        is_same = sender_species == receiver_species  # It is from the same species
        decided_correct = decided_same == is_same  # The decision was correct

        if not is_same:
            return None

        if null:
            fitness = 3 * np.product([nonlin_fitness(1 - abs(o - d)) for o, d in zip(original, decode)])
            fitness = correct_multiplier(fitness, original, decode)
            return fitness

        # How close did the decoder come to determining the correct species
        fitness = nonlin_fitness(1 - abs(int(is_same) - decode[-1]))

        if decided_correct:
            fitness += 3*np.product([nonlin_fitness(1-abs(o - d)) for o, d in zip(original, decode)])
            fitness = correct_multiplier(fitness, original, decode)

        return fitness

    @staticmethod
    def receiver_fitness(original, decode, sender_species, receiver_species, null=False):
        '''Score the receiver.

        The receiver is scored similarly to the sender, but it also gets points for correctly identified non-member
        calls.

        Maximum theoretical fitness is 4 * n_messages * n_individuals + n_messages * conspecific_individuals
        = 4*7*50 + 7*50 = 1400 + 350 = 1750

        With the nonlinearity, the actual theoretical maximum (assuming decode on [0,1]) is 0.99966464987 for each bit.
        This works out to 2.99698286085 once you take the product of all three bits and multiply by 3.
        With the multiplication factors (1.0+0.1n) for correct bits and the score for the correct species, this works
        out to a possible total of 6.85882258922 points per message.

        :todo Do I need to bump the scores (add 1.5, for example) for correctly identified non-members?

        :param original:
        :param decode:
        :param sender_species:
        :param receiver_species:
        :return:
        '''
        decided_same = decode[-1] >= 0.5  # NN thinks its from the same species
        is_same = sender_species == receiver_species  # It is from the same species
        decided_correct = decided_same == is_same  # The decision was correct

        if null:
            if not is_same:
                return None

            fitness = 3 * np.product([nonlin_fitness(1 - abs(o - d)) for o, d in zip(original, decode)])
            fitness = correct_multiplier(fitness, original, decode)
            return fitness

        fitness = nonlin_fitness(1 - abs(int(is_same) - decode[-1]))

        if decided_correct and is_same:
            fitness += 3*np.product([nonlin_fitness(1 - abs(o - d)) for o, d in zip(original, decode)])
            fitness = correct_multiplier(fitness, original, decode)

        return fitness


    @staticmethod
    def score(original, decode, sender_species, receiver_species):
        '''Finds the score for a decoded message.

        This score is not the fitness. It is an external measure of the performance of the decoder in terms of the
        percentage of messages that it decodes correctly. The score is returned as a tuple/list in the form of
        ``[ species, bits, total ]``.

        ``species`` is a simple binary measuring whether the decoder correctly identified the sample as coming from
        within the group of outside of it.
        ``bit`` measures the proportion of bits that are correctly categorized. This looks only at messages from within
        the species and assigns an automatic 0 id the species is incorrectly identified.
        ``total`` measures the proportion of messages where the entire message is correctly decoded. It looks only at
        messages from within the species and assigns an automatic 0 id the species is incorrectly identified.

        :param original: Original message, 3 bit array, assumed to be on [0..1]
        :param decode: Decoded message, 4 bit array
        :param sender_species: ID of the sender species
        :param receiver_species: ID of the receiver species
        :return:
        '''

        decided_same = decode[-1] >= 0.5  # NN thinks its from the same species
        is_same = sender_species == receiver_species # It is from the same species
        decided_correct = decided_same == is_same  # The decision was correct

        # Score the species selection
        species = int(decided_correct)

        if is_same and not decided_correct:
            bits = 0
            total = 0
        elif is_same:
            bits = 3 - sum([abs(o-round(d)) for o,d in zip(original, decode)])
            total = 1.0 if bits == 3 else 0.0
            bits /= 3.0
        else:
            bits = None
            total = None

        return [species, bits, total]


class PairwiseDecoderEvaluator(BaseEvaluator):
    '''A near-clone of DecoderEvaluator, but this version evaluates each encoder/decoder on only a single
    encoder/decoder from its species.

    Only works for two species for now...
    '''

    def evaluator(self, genomes, config):
        super(PairwiseDecoderEvaluator, self).evaluator(genomes, config)

        genomes = dict(genomes)
        genome_ids = list(genomes.keys())
        eval_count = dict([(g, 0) for g in genome_ids])
        pairing = {}
        pair_count = dict([(g, 0) for g in genome_ids])

        encoder_fitness = {}
        species_scores = {}
        bit_scores = {}
        total_scores = {}
        species_ids = []
        false_count = 0

        # Wait for encoded messages from encoders
        encoded_tuple = self.encoded.get()  # Assume the first one isn't false
        while True:
            enc_species_id, enc_genome_id, original_message, encoded_message = encoded_tuple

            # Update available species IDs, test for completeness
            if enc_species_id not in species_ids:
                species_ids.append(enc_species_id)

            is_same = enc_species_id == self.species_id  # It is from the same species

            if enc_genome_id not in encoder_fitness and is_same:
                encoder_fitness[enc_genome_id] = [0, 0]  # Score, count

            if enc_genome_id not in pairing:
                least = min(pair_count.values())
                options = filter(lambda k: pair_count[k] == least, pair_count)

                genome_id = choice(list(options))
                pairing[enc_genome_id] = genome_id
                pair_count[genome_id] += 1

            genome_id = pairing[enc_genome_id]
            eval_count[genome_id] += 1

            genome = genomes[genome_id]
            if genome_id not in species_scores:
                species_scores[genome_id] = []
                bit_scores[genome_id] = []
                total_scores[genome_id] = []

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            decoded_message = net.activate(encoded_message)

            ''' Score the decoding.
            First, check to see whether it got the species identification (last bit) correct. This should be 1
            if it is the same species, 0 otherwise. The penalty for incorrect identification is -4.

            If species ID is correct AND this is from one's own species:
            The score is the absolute difference between the floating point (0..1) output of the 
            decoder and the boolean (0/1) input of the original message.

            If this is a multispecies optimization, an additional condition is occurs. The messages should be
            treated as neutral or unimportant and any message from another species is scored as the absolute
            difference from an array of 0.5. Scores are only recorded for the encoders of the same species.
            OLD!

            There are also three additional decoder scores. Species score, bit score, and total score. Species score
            is 1/-1 whether it correctly identified its own/other species. Bit score is how many of 3 bits were
            correct (if species was a match and identified correctly). Total score is 1 point if species was
            correctly identified as a match and all three bits were correct.
            '''

            decided_same = decoded_message[-1] > 0  # NN thinks its from the same species
            is_same = enc_species_id == self.species_id  # It is from the same species
            decided_correct = decided_same == is_same  # The decision was correct
            fitness = (1 + (1 if decided_correct else -1) * tanh(abs(decoded_message[-1]))) / 2

            if decided_correct:  # The species and species prediction match
                species_scores[genome_id].append(1)
            else:
                species_scores[genome_id].append(0)

            if is_same:  # This is the same species
                if decided_same:
                    bit_score = 6 - sum([abs(o - round(d)) for o, d in zip(original_message, decoded_message)])
                    bit_scores[genome_id].append(bit_score / 6)
                    total_scores[genome_id].append(bit_score == 6 and 1 or 0)

                    fitness += (6 - sum([abs(o - d) for o, d in zip(original_message, decoded_message)])) / 2

                # Pass the score back for the encoder
                encoder_fitness[enc_genome_id][0] += fitness
                encoder_fitness[enc_genome_id][1] += 1  # Count

            # Register the score for the genome
            genome.fitness += fitness

            encoded_tuple = self.encoded.get()
            do_break = False
            while encoded_tuple is False:
                false_count += 1
                if false_count == len(species_ids):
                    print('Stopped with %i False values' % false_count)
                    print(species_ids)
                    do_break = True
                    break
                encoded_tuple = self.encoded.get()
            if do_break:
                break

        for id, genome in genomes.items():
            try:
                genome.fitness /= eval_count[id]
            except ZeroDivisionError as e:
                print('Divide by zero on Genome {}'.format(id))

        encoder = {}
        for id, (total_fitness, count) in encoder_fitness.items():
            encoder[id] = total_fitness / count

        self.scores.put(encoder)

        self.genomes.put(genomes)

        self.decoding_scores.put({'species': species_scores, 'bit': bit_scores, 'total': total_scores})

if __name__ == "__main__":
    import doctest
    doctest.testmod()