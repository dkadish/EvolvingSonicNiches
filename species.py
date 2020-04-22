import logging
import os
import random
from datetime import datetime
from multiprocessing import Queue
from threading import Thread

from natsort import natsorted

import neat
from evaluators import EncoderEvaluator, DecoderEvaluator
from parallel import MultiQueue
from population import Population, restore_checkpoint
from stats import DataFrameReporter

logger = logging.getLogger('evolvingniches.species')

class CommunicatorSet:

    def __init__(self, population: neat.Population, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir

        self.population = population

        self.stats = neat.StatisticsReporter()

        # Set up Queues
        self.genomes = Queue()

        # Set up reporters
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)

        self.evaluator = None
        self.df_reporter = None
        self.thread = None

    @property
    def config(self):
        return self.population.config

    @classmethod
    def create(cls, config, checkpoint_dir=None):
        population = Population(config)

        return cls(population, checkpoint_dir=checkpoint_dir)

    @classmethod
    def load(cls, checkpoint_file, checkpoint_dir):
        # NEAT Checkpointer reloads the random state from the previous run, meaning that each new run will behave
        # the same way. We don't want that, so we save the current random state to re-load after we've restored the
        # checkpoints
        random_state = random.getstate()
        pop = restore_checkpoint(checkpoint_file)
        random.setstate(random_state)
        pop.generation += 1

        return cls(pop, checkpoint_dir=checkpoint_dir)


class EncoderSet(CommunicatorSet):

    @classmethod
    def create(cls, config, checkpoint_dir, messages, scores,
               decoding_scores, species_id, run_id, evaluator_config):
        s = super().create(config, checkpoint_dir)

        return cls.__finish_setup(decoding_scores, evaluator_config, messages, run_id, s, scores, species_id)

    @classmethod
    def load(cls, checkpoint_file, checkpoint_dir, messages, scores,
             decoding_scores, species_id, run_id, evaluator_config):

        s = super().load(checkpoint_file, checkpoint_dir)

        return cls.__finish_setup(decoding_scores, evaluator_config, messages, run_id, s, scores, species_id)

    @classmethod
    def __finish_setup(cls, decoding_scores, evaluator_config, messages, run_id, s, scores, species_id):
        if evaluator_config is not None:
            s.evaluator = EncoderEvaluator(
                messages, scores, s.genomes,
                decoding_scores, species_id, run_id, config=evaluator_config)
        else:
            s.evaluator = EncoderEvaluator(
                messages, scores, s.genomes,
                decoding_scores, species_id, run_id)

        if s.checkpoint_dir is not None:
            prefix = '{}/{:%y-%m-%d-%H-%M-%S}_neat-enc-checkpoint-'.format(s.checkpoint_dir, datetime.now())
        else:
            prefix = '{:%y-%m-%d-%H-%M-%S}_neat-enc-checkpoint-'.format(datetime.now())

        # NB: Reporters are not saved with the population state.
        checkpointer = neat.Checkpointer(generation_interval=100, time_interval_seconds=None, filename_prefix=prefix)
        if s.population.generation > 0:
            checkpointer.last_generation_checkpoint = s.population.generation - 1 # We added one to the population when we loaded it.
        s.population.add_reporter(checkpointer)
        s.df_reporter = DataFrameReporter(run=run_id, species=species_id, role='sender')
        s.population.add_reporter(s.df_reporter)
        return s


class DecoderSet(CommunicatorSet):
    @classmethod
    def create(cls, config, checkpoint_dir, messages, scores,
               decoding_scores, species_id, run_id, evaluator_config):
        s = super().create(config, checkpoint_dir)

        return cls.__finish_setup(decoding_scores, evaluator_config, messages, run_id, s, scores, species_id)

    @classmethod
    def load(cls, checkpoint_file, checkpoint_dir, messages, scores,
             decoding_scores, species_id, run_id, evaluator_config):

        s = super().load(checkpoint_file, checkpoint_dir)

        return cls.__finish_setup(decoding_scores, evaluator_config, messages, run_id, s, scores, species_id)

    @classmethod
    def __finish_setup(cls, decoding_scores, evaluator_config, messages, run_id, s, scores, species_id):
        if evaluator_config is not None:
            s.evaluator = DecoderEvaluator(
                messages, scores, s.genomes,
                decoding_scores, species_id, run_id, config=evaluator_config)
        else:
            s.evaluator = DecoderEvaluator(
                messages, scores, s.genomes,
                decoding_scores, species_id, run_id)
        if s.checkpoint_dir is not None:
            prefix = '{}/{:%y-%m-%d-%H-%M-%S}_neat-dec-checkpoint-'.format(s.checkpoint_dir, datetime.now())
        else:
            prefix = '{:%y-%m-%d-%H-%M-%S}_neat-dec-checkpoint-'.format(datetime.now())

        checkpointer = neat.Checkpointer(generation_interval=100, time_interval_seconds=None, filename_prefix=prefix)
        if s.population.generation > 0:
            checkpointer.last_generation_checkpoint = s.population.generation - 1 # We added one to the population when we loaded it.
        s.population.add_reporter(checkpointer)
        s.df_reporter = DataFrameReporter(run=run_id, species=species_id, role='receiver')
        s.population.add_reporter(s.df_reporter)
        return s


class Species:
    counter = 0

    def __init__(self, messages: MultiQueue, run=0):
        self.messages = messages
        self.scores = Queue()
        self.dataframe_list = Queue()

        self.species_id = Species.counter
        Species.counter += 1

        self.run_id = run

        self.encoder = None
        self.decoder = None

    def start(self):

        self.encoder.thread = Thread(target=self.encoder.population.run_generation,
                                     args=(self.encoder.evaluator.evaluate,))
        self.decoder.thread = Thread(target=self.decoder.population.run_generation,
                                     args=(self.decoder.evaluator.evaluate,))

        logger.debug('New species id: %i' % self.species_id)

        self.encoder.thread.start()
        self.decoder.thread.start()

    def join(self, timeout=None):
        if timeout is None:
            self.encoder.thread.join()
            self.decoder.thread.join()
        else:
            self.encoder.thread.join(timeout / 2)
            self.decoder.thread.join(timeout / 2)

    def is_alive(self):
        return self.encoder.thread.is_alive() or self.decoder.thread.is_alive()

    @property
    def generation(self):
        assert self.encoder.population.generation == self.decoder.population.generation

        return self.encoder.population.generation

    @staticmethod
    def create(encoder_config, decoder_config, messages: MultiQueue, run=0, checkpoint_dir=None, evaluator_config=None):
        s = Species(messages, run)

        #TODO Do we need to messages.add() here too? If not, why?
        s.encoder = EncoderSet.create(encoder_config, checkpoint_dir, s.messages, s.scores,
                                      s.dataframe_list, s.species_id, s.run_id, evaluator_config)

        s.decoder = DecoderSet.create(decoder_config, checkpoint_dir, s.messages.add(), s.scores,
                                      s.dataframe_list, s.species_id, s.run_id, evaluator_config)

        return s

    @staticmethod
    def load(directory, messages: MultiQueue, run=0, checkpoint_dir=None, evaluator_config=None):
        s = Species(messages, run)

        logger.debug('Loading checkpoints from {}'.format(os.path.abspath(directory)))
        checkpoint_files = natsorted(filter(lambda f: 'checkpoint' in f, os.listdir(directory)))
        enc_file = os.path.join(directory, list(filter(lambda f: 'enc' in f, checkpoint_files))[-1])
        dec_file = os.path.join(directory, list(filter(lambda f: 'dec' in f, checkpoint_files))[-1])

        #TODO Do we need to messages.add() here too? If not, why?
        s.encoder = EncoderSet.load(enc_file, checkpoint_dir, s.messages, s.scores,
                                    s.dataframe_list, s.species_id, s.run_id, evaluator_config)

        s.decoder = DecoderSet.load(dec_file, checkpoint_dir, s.messages.add(), s.scores,
                                    s.dataframe_list, s.species_id, s.run_id, evaluator_config)

        return s
