from multiprocessing import Queue
from threading import Thread

import neat
from evaluators import EncoderEvaluator, DecoderEvaluator
from parallel import MultiQueue



class CommunicatorSet:

    def __init__(self, config):
        self.config = config
        self.population = neat.Population(config)
        self.stats = neat.StatisticsReporter()

        # Set up Queues
        self.genomes = Queue()

        # Set up reporters
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)
        self.population.add_reporter(neat.Checkpointer(5))

        self.evaluator = None

        self.thread = None

    def createEncoderEvaluator(self, encoded, scores, spectra, cohesion, species_id):
        self.evaluator = EncoderEvaluator(encoded, scores, self.genomes, spectra, cohesion, species_id)

    def createDecoderEvaluator(self, encoded, scores, spectra, cohesion, species_id):
        self.evaluator = DecoderEvaluator(encoded, scores, self.genomes, spectra, cohesion, species_id)


class Species:

    counter = 0

    def __init__(self, encoder_config, decoder_config, encoded: MultiQueue):
        self.encoded = encoded
        self.scores = Queue()
        self.spectra = Queue()
        self.cohesion = Queue()

        self.species_id = Species.counter
        Species.counter += 1

        self.encoder = CommunicatorSet(encoder_config)
        self.encoder.createEncoderEvaluator(self.encoded, self.scores, self.spectra, self.cohesion, self.species_id)

        self.decoder = CommunicatorSet(decoder_config)
        self.decoder.createDecoderEvaluator(self.encoded.add(), self.scores, self.spectra, self.cohesion, self.species_id)

        print('New species id: %i' %self.species_id)

    def start(self):

        self.encoder.thread = Thread(target=self.encoder.population.run_generation, args=(self.encoder.evaluator.evaluate,))
        self.decoder.thread = Thread(target=self.decoder.population.run_generation, args=(self.decoder.evaluator.evaluate,))

        self.encoder.thread.start()
        self.decoder.thread.start()

    def join(self):
        self.encoder.thread.join()
        self.decoder.thread.join()
