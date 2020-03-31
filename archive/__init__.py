import joblib

import neat
from archive.filterable import Filterable
from archive.fitness import FitnessList
from archive.messages import MessageList
from archive.score import ScoreList


class Spectrum(Filterable):

    def __init__(self, run=None, generation=None, species=None, subspecies=None):
        super().__init__(run, generation, species, subspecies)


class EnvironmentConfig:
    class Simulation:

        def __init__(self, n_messages=7, noise_overwrites_signal=False, n_generations=300,
                     noise_channel=[4], noise_level=[1.0]):
            self.n_messages = n_messages
            self.noise_overwrites_signal = noise_overwrites_signal
            self.n_generations = n_generations
            self.noise_channel = noise_channel
            self.noise_level = noise_level

    class Evaluation:

        def __init__(self, correct_factor=0.1, loudness_penalty=0.1, no_species_id_score=True):
            self.correct_factor = correct_factor
            self.loudness_penalty = loudness_penalty
            self.no_species_id_score = no_species_id_score


class Config:

    def __init__(self, sender: neat.config.Config, receiver: neat.config.Config, environment: EnvironmentConfig):
        self.sender = sender
        self.receiver = receiver
        self.environment = environment


class Archive:

    def __init__(self, messages=[], fitnesses=[], scores=[], configs:Config=None, **kwargs) -> None:
        self.messages = MessageList(messages)
        self.fitness = FitnessList(fitnesses)
        self.scores = ScoreList(scores)

    def add_run(self, message_list, fitness_list, score_list, run_id=None):
        # if run_id is None:
        #     run_id = self.messages.next_run
        #
        # ml = MessageList.from_message_archive(message_archive, run=run_id)

        self.messages.extend(message_list)
        self.fitness.extend(fitness_list)
        self.scores.extend(score_list)

    def save(self, filename):
        joblib.dump(self, filename)

    def __add__(self, other):
        raise NotImplementedError('Not Implemented yet.')

    @staticmethod
    def load(filename):
        return joblib.load(filename)

    @staticmethod
    def createArchive(message_archive):
        pass
