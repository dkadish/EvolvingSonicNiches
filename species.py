from datetime import datetime
from multiprocessing import Queue
from threading import Thread

import neat
from evaluators import EncoderEvaluator, DecoderEvaluator, PairwiseDecoderEvaluator
from parallel import MultiQueue
from population import Population


class CommunicatorSet:

    def __init__(self, config, checkpoint_dir=None):
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

        self.checkpoint_dir = checkpoint_dir

    def createDecoderEvaluator(self,
                               messages, scores,
                               decoding_scores, species_id, run_id, config):
        if config is not None:
            self.evaluator = DecoderEvaluator(
                                          messages, scores, self.genomes,
                                          decoding_scores, species_id, run_id, config=config)
        else:
            self.evaluator = DecoderEvaluator(
                messages, scores, self.genomes,
                decoding_scores, species_id, run_id)
        
        if self.checkpoint_dir is not None:
            prefix = '{}/{:%y-%m-%d-%H-%M-%S}_neat-dec-checkpoint-'.format(self.checkpoint_dir,datetime.now())
        else:
            prefix='{:%y-%m-%d-%H-%M-%S}_neat-dec-checkpoint-'.format(datetime.now())
        self.population.add_reporter(neat.Checkpointer(5, filename_prefix=prefix))

    def createEncoderEvaluator(self,
                               messages, scores,
                               decoding_scores, species_id, run_id, config):
        if config is not None:
            self.evaluator = EncoderEvaluator(
                                          messages, scores, self.genomes,
                                          decoding_scores, species_id, run_id, config=config)
        else:
            self.evaluator = EncoderEvaluator(
                                          messages, scores, self.genomes,
                                          decoding_scores, species_id, run_id)
        if self.checkpoint_dir is not None:
            prefix = '{}/{:%y-%m-%d-%H-%M-%S}_neat-enc-checkpoint-'.format(self.checkpoint_dir,datetime.now())
        else:
            prefix='{:%y-%m-%d-%H-%M-%S}_neat-enc-checkpoint-'.format(datetime.now())

        self.population.add_reporter(neat.Checkpointer(5, filename_prefix=prefix))

    def createPairwireDecoderEvaluator(self, #encoded,
                                       messages, scores,
                                       decoding_scores, species_id):
        self.evaluator = PairwiseDecoderEvaluator(#encoded,
                                                  messages, scores, self.genomes,
                                                  decoding_scores, species_id)


class Species:

    counter = 0

    def __init__(self, encoder_config, decoder_config, #encoded: MultiQueue,
                 messages: MultiQueue, run=0, pairwise=False, checkpoint_dir=None, evaluator_config=None):
        # self.encoded = encoded
        self.messages = messages
        self.scores = Queue()
        self.dataframe_list = Queue()

        self.species_id = Species.counter
        Species.counter += 1

        self.run_id = run

        self.encoder = CommunicatorSet(encoder_config, checkpoint_dir)
        self.encoder.createEncoderEvaluator(
                                            self.messages, self.scores,
                                            self.dataframe_list, self.species_id, self.run_id, evaluator_config)

        if not pairwise:
            self.decoder = CommunicatorSet(decoder_config, checkpoint_dir)
            self.decoder.createDecoderEvaluator(
                                                self.messages.add(), self.scores,
                                                self.dataframe_list, self.species_id, self.run_id, evaluator_config)
        else:
            self.decoder = CommunicatorSet(decoder_config, checkpoint_dir)
            self.decoder.createPairwireDecoderEvaluator(
                                                        self.messages.add(), self.scores,
                                                        self.dataframe_list, self.species_id, self.run_id, evaluator_config)

        print('New species id: %i' %self.species_id)

    def start(self):

        self.encoder.thread = Thread(target=self.encoder.population.run_generation, args=(self.encoder.evaluator.evaluate,))
        self.decoder.thread = Thread(target=self.decoder.population.run_generation, args=(self.decoder.evaluator.evaluate,))

        self.encoder.thread.start()
        self.decoder.thread.start()

    def join(self, timeout=None):
        if timeout is None:
            self.encoder.thread.join()
            self.decoder.thread.join()
        else:
            self.encoder.thread.join(timeout/2)
            self.decoder.thread.join(timeout/2)

    def is_alive(self):
        return self.encoder.thread.is_alive() or self.decoder.thread.is_alive()
