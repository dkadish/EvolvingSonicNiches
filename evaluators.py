import random
from multiprocessing import Queue, Process

import neat

N_MESSAGES = 10


class BaseEvaluator:

    def __init__(self, encoded: Queue, scores: Queue, genomes: Queue):
        # self.messages = messages

        # self.encoded Format: (genome id, original message, encoded message)
        self.encoded = encoded

        # self.scores Format: (genome id, score)
        self.scores = scores
        self.genomes = genomes

        self.process = None

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
        messages = [[[int(random.random() > 0.5) for i in range(3)] for j in range(N_MESSAGES)] for k in
                    range(len(genomes))]

        # Create the NNs and encode the messages
        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for original_message in messages[i]:
                encoded_message = net.activate(original_message)
                self.encoded.put((genome_id, original_message, encoded_message))

        # Send Finished Message
        self.encoded.put(False)

        # Wait for the scores from the decoders and evaluate
        scores = self.scores.get()

        genome_dict = dict(genomes)
        for genome_id in scores:
            genome_dict[genome_id].fitness = scores[genome_id]

        self.genomes.put(genome_dict)


class DecoderEvaluator(BaseEvaluator):

    def evaluator(self, genomes, config):
        super(DecoderEvaluator, self).evaluator(genomes, config)

        encoder_scores = {}

        # Wait for encoded messages from encoders
        encoded_tuple = self.encoded.get()
        while encoded_tuple is not False:
            enc_genome_id, original_message, encoded_message = encoded_tuple
            if enc_genome_id not in encoder_scores:
                encoder_scores[enc_genome_id] = 0

            for genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                decoded_message = net.activate(encoded_message)
                score = -sum([abs(o - d) for o, d in zip(original_message, decoded_message)])

                # Pass the score back for the encoder
                encoder_scores[enc_genome_id] += score
                # self.scores.put((genome_id, score))

                # Register the score for the genome
                if genome.fitness is None:
                    genome.fitness = score
                else:
                    genome.fitness += score

            encoded_tuple = self.encoded.get()

        self.scores.put(encoder_scores)

        self.genomes.put(dict(genomes))
