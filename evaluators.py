import random
from multiprocessing import Queue

import numpy as np
from scipy.spatial.distance import cdist

import neat

N_MESSAGES = 10


class BaseEvaluator:

    def __init__(self, encoded: Queue, scores: Queue, genomes: Queue, spectra: Queue, cohesion: Queue, species_id=0):
        # self.messages = messages

        # self.encoded Format: (genome id, original message, encoded message)
        self.encoded = encoded

        # self.scores Format: (genome id, score)
        self.scores = scores
        self.genomes = genomes

        self.process = None

        self.spectra = spectra
        self.cohesion = cohesion

        self.species_id = species_id  # For identifying members of the same species in multispecies processes.

    def __del__(self):
        pass

    def evaluate(self, genomes, config):
        self.evaluator(genomes, config)

        g = self.genomes.get()
        for genome_id, genome in genomes:
            genome.fitness = g[genome_id].fitness

    def evaluator(self, genomes, config):
        pass


class EncoderEvaluator(BaseEvaluator):

    def evaluator(self, genomes, config):
        super(EncoderEvaluator, self).evaluator(genomes, config)

        # Generate a set of messages to use for testing
        messages = [[tuple([int(random.random() > 0.5) for i in range(3)]) for j in range(N_MESSAGES)] for k in
                    range(len(genomes))]

        # Save spectral information about messages
        spectra = None
        message_cohesion = {}

        # Create the NNs and encode the messages
        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            if spectra is None:
                spectra = [0 for s in range(len(net.output_nodes))]
            for original_message in messages[i]:
                encoded_message = net.activate(original_message)
                spectra = [s+e for s, e in zip(spectra, encoded_message)]
                if original_message not in message_cohesion:
                    message_cohesion[original_message] = []
                message_cohesion[original_message].append(encoded_message)
                self.encoded.put((self.species_id, genome_id, original_message, encoded_message))

        # Send Finished Message
        self.encoded.put(False)

        # Send the spectra back
        self.spectra.put(spectra)

        # Evaluate the distance between messages
        #TODO This should be implemented somewhere else that listens to the environment, using the multiqueue.
        mc_score = {}
        for m in message_cohesion:
            mc = np.array(message_cohesion[m])
            distances = cdist(mc, mc)

            mc_score[m] = np.average(distances)
        self.cohesion.put(mc_score)

        # Wait for the scores from the decoders and evaluate
        scores = self.scores.get()

        genome_dict = dict(genomes)
        for genome_id in scores:
            try:
                genome_dict[genome_id].fitness = scores[genome_id]
            except KeyError as e:
                pass

        self.genomes.put(genome_dict)


class DecoderEvaluator(BaseEvaluator):

    def evaluator(self, genomes, config):
        super(DecoderEvaluator, self).evaluator(genomes, config)

        encoder_scores = {}
        species_ids = []
        false_count = 0

        # Wait for encoded messages from encoders
        encoded_tuple = self.encoded.get() # Assume the first one isn't false
        while True:
            enc_species_id, enc_genome_id, original_message, encoded_message = encoded_tuple

            # Update available species IDs, test for completeness
            if enc_species_id not in species_ids:
                species_ids.append(enc_species_id)

            if enc_genome_id not in encoder_scores and enc_species_id == self.species_id:
                encoder_scores[enc_genome_id] = 0

            for genome_id, genome in genomes:
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
                '''

                same_species_code = bool(round(decoded_message[-1])) # Is this from my species?
                if same_species_code == (enc_species_id == self.species_id): # The species and species prediction match
                    score = 0
                else:
                    score = -4

                if enc_species_id == self.species_id: # This is the same species
                    if same_species_code:
                        score = -sum([abs(o - round(d)) for o, d in zip(original_message, decoded_message)])

                    # Pass the score back for the encoder
                    encoder_scores[enc_genome_id] += score
                # else:
                #     score = -sum([abs(0 - round(d)) for d in decoded_message]) / 3.0

                # Register the score for the genome
                if genome.fitness is None:
                    genome.fitness = score
                else:
                    genome.fitness += score

            encoded_tuple = self.encoded.get()
            do_break = False
            while encoded_tuple is False:
                false_count += 1
                if false_count == len(species_ids):
                    print('Stopped with %i False values' %false_count)
                    print(species_ids)
                    do_break = True
                    break
                encoded_tuple = self.encoded.get()
            if do_break:
                break

        self.scores.put(encoder_scores)

        self.genomes.put(dict(genomes))
