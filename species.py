from datetime import datetime
from multiprocessing import Queue
from threading import Thread

import neat
from evaluators import EncoderEvaluator, DecoderEvaluator, PairwiseDecoderEvaluator
from parallel import MultiQueue
from population import Population


class CommunicatorSet:

    def __init__(self, config):
        self.config = config
        self.population = Population(config)
        self.stats = neat.StatisticsReporter()

        # Set up Queues
        self.genomes = Queue()

        # Set up reporters
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)
        # self.population.add_reporter(neat.Checkpointer(5))

        self.evaluator = None

        self.thread = None

    def createDecoderEvaluator(self, #encoded,
                               messages, scores,
                               decoding_scores, species_id):
        self.evaluator = DecoderEvaluator(#encoded,
                                          messages, scores, self.genomes,
                                          decoding_scores, species_id)

        self.population.add_reporter(neat.Checkpointer(5, filename_prefix='{:%y-%m-%d-%H-%M-%S}_neat-dec-checkpoint-'.format(datetime.now())))

    def createEncoderEvaluator(self, #encoded,
                               messages, scores,
                               decoding_scores, species_id):
        self.evaluator = EncoderEvaluator(#encoded,
                                          messages, scores, self.genomes,
                                          decoding_scores, species_id)
        self.population.add_reporter(neat.Checkpointer(5, filename_prefix='{:%y-%m-%d-%H-%M-%S}_neat-enc-checkpoint-'.format(datetime.now())))

    def createPairwireDecoderEvaluator(self, #encoded,
                                       messages, scores,
                                       decoding_scores, species_id):
        self.evaluator = PairwiseDecoderEvaluator(#encoded,
                                                  messages, scores, self.genomes,
                                                  decoding_scores, species_id)


class Species:

    counter = 0

    def __init__(self, encoder_config, decoder_config, #encoded: MultiQueue,
                 messages: MultiQueue, pairwise=False):
        # self.encoded = encoded
        self.messages = messages
        self.scores = Queue()
        self.decoding_scores = Queue()

        self.species_id = Species.counter
        Species.counter += 1

        self.encoder = CommunicatorSet(encoder_config)
        self.encoder.createEncoderEvaluator(#self.encoded,
                                            self.messages, self.scores,
                                            self.decoding_scores, self.species_id)

        if not pairwise:
            self.decoder = CommunicatorSet(decoder_config)
            self.decoder.createDecoderEvaluator(#self.encoded.add(),
                                                self.messages.add(), self.scores,
                                                self.decoding_scores, self.species_id)
        else:
            self.decoder = CommunicatorSet(decoder_config)
            self.decoder.createPairwireDecoderEvaluator(#self.encoded.add(),
                                                        self.messages.add(), self.scores,
                                                        self.decoding_scores, self.species_id)

        print('New species id: %i' %self.species_id)

    def start(self):

        self.encoder.thread = Thread(target=self.encoder.population.run_generation, args=(self.encoder.evaluator.evaluate,))
        self.decoder.thread = Thread(target=self.decoder.population.run_generation, args=(self.decoder.evaluator.evaluate,))

        self.encoder.thread.start()
        self.decoder.thread.start()

    def join(self):
        self.encoder.thread.join()
        self.decoder.thread.join()
