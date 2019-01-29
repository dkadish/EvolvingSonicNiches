import random
from math import tanh
from multiprocessing import Queue
from random import choice

import numpy as np
from scipy.spatial.distance import cdist

import neat
from messaging import Message

N_MESSAGES = 10


class BaseEvaluator:

    def __init__(self, encoded: Queue, messages: Queue, scores: Queue, genomes: Queue, spectra: Queue, cohesion: Queue,
                 decoding_scores: Queue, species_id=0):
        # self.messages = messages

        # self.encoded Format: (genome id, original message, encoded message)
        self.encoded = encoded
        self.messages = messages

        # self.scores Format: (genome id, score)
        self.scores = scores
        self.genomes = genomes

        self.process = None

        self.spectra = spectra
        self.cohesion = cohesion
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

    def evaluator(self, genomes, config):
        super(EncoderEvaluator, self).evaluator(genomes, config)

        # Generate a set of messages to use for testing
        messages = [[tuple([int(random.random() > 0.5) for i in range(3)]) for j in range(N_MESSAGES)] for k in
                    range(len(genomes))]

        # Save spectral information about messages
        # spectra = None
        message_loudness = []
        message_cohesion = {}

        # Create the NNs and encode the messages
        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # if spectra is None:
            #     spectra = [0 for s in range(len(net.output_nodes))]
            for original_message in messages[i]:
                encoded_message = net.activate(original_message)
                message_loudness.extend(encoded_message)
                # spectra = [s + e for s, e in zip(spectra, encoded_message)]
                if original_message not in message_cohesion:
                    message_cohesion[original_message] = []
                message_cohesion[original_message].append(encoded_message)
                self.encoded.put((self.species_id, genome_id, original_message, encoded_message))
                self.messages.put(Message.Encoded(self.species_id, genome_id, original_message, encoded_message))

        # Send Finished Message
        self.encoded.put(False)
        self.messages.put(Message.Generation(self.species_id))

        # Send the spectra back
        # self.spectra.put(spectra)

        # Evaluate the distance between messages
        # TODO This should be implemented somewhere else that listens to the environment, using the multiqueue.
        mc_score = {}
        mc_loudness = {}  # FIXME This has nothing to do with message cohesion
        for m in message_cohesion:
            mc = np.array(message_cohesion[m])
            distances = cdist(mc, mc)

            mc_score[m] = np.average(distances)

        mc_loudness['avg'] = np.average(np.abs(message_loudness))
        mc_loudness['std'] = np.std(np.abs(message_loudness))

        self.cohesion.put([mc_score, mc_loudness])

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
        encoded_tuple = self.encoded.get()  # Assume the first one isn't false
        while True:
            enc_species_id, enc_genome_id, original_message, encoded_message = encoded_tuple

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

                # Ensure decoded_message is between 0 and 1
                decoded_message = np.clip(decoded_message, 0, 1)

                decided_same = decoded_message[-1] >= 0.5  # NN thinks its from the same species
                is_same = enc_species_id == self.species_id  # It is from the same species
                decided_correct = decided_same == is_same  # The decision was correct
                fitness = 1 - abs(int(is_same) - decoded_message[-1])

                if decided_correct:  # The species and species prediction match
                    species_scores[genome_id].append(1)
                else:
                    species_scores[genome_id].append(0)

                if is_same:  # This is the same species
                    if decided_same:
                        bit_score = 3 - sum([abs(o - round(d)) for o, d in zip(original_message, decoded_message)])
                        bit_scores[genome_id].append(bit_score / 3)
                        total_scores[genome_id].append(bit_score == 3 and 1 or 0)

                        fitness += (3 - sum([abs(o - d) for o, d in zip(original_message, decoded_message)]))

                    # Pass the score back for the encoder
                    encoder_fitness[enc_genome_id] += fitness

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

        self.scores.put(encoder_fitness)

        self.genomes.put(dict(genomes))

        self.decoding_scores.put({'species': species_scores, 'bit': bit_scores, 'total': total_scores})


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
