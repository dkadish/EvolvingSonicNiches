import itertools
import unittest

import numpy as np

from archive import MessageList, Filterable
from archive.filterable import FilterableList
from archive.fitness import Fitness, FitnessList
from archive.messages import Messages
from archive.score import Score, ScoreList
from neat import DefaultGenome, StatisticsReporter
from evolvingniches import stats


class TestArchiveFilterableList(unittest.TestCase):

    def test_filterableList_addition(self):
        permutations = sorted(list(set(itertools.permutations([0, 0, 0, 0, 1, 1, 1, 1], 4))))[1:]
        fl = FilterableList([Filterable(run=r, generation=g, species=s, subspecies=ss) for r, g, s, ss in permutations])
        permutations2 = sorted(list(set(itertools.permutations([0, 0, 1, 1, 2, 2, 3, 3], 4))))[1:]
        fl2 = FilterableList([Filterable(run=r, generation=g, species=s, subspecies=ss) for r, g, s, ss in permutations2])
        fl3 = fl+fl2

        fl4 = FilterableList(fl.data)
        assert fl4.count == fl.count
        fl4.extend(fl2)

        assert fl.count == 15
        assert fl2.count == 203
        assert fl3.count == 218
        assert fl4.count == 218

        assert fl3.filter(run=1).count == 59
        assert fl3.filter(generation=2).count == 51

        assert fl4.filter(run=1).count == 59
        assert fl4.filter(generation=2).count == 51

class TestArchiveMessages(unittest.TestCase):

    def _generate_random_messages(self):
        original = np.random.choice([0, 1], size=(350, 3))
        encoded = np.random.random(size=(350, 9))
        noise = np.zeros(shape=(350, 9))
        noise[:, 4] += 1.0
        received = encoded + noise
        messages = Messages(original, encoded, received, run=0, generation=0, species=0, subspecies=0)

        return messages

    def _generate_ordered_messages(self, run=0, generation=0, species=0, subspecies=0):
        original_set = sorted(list(set(itertools.permutations([0, 0, 0, 1, 1, 1], 3))))[1:]
        original = np.concatenate([np.repeat([o], 50, axis=0) for o in original_set])
        encoded = np.average(np.mgrid[0:1:350j, 0:1:9j], axis=0)
        noise = np.zeros(shape=(350, 9))
        noise[:, 4] += 1.0
        received = encoded + noise
        messages = Messages(original, encoded, received, run=run, generation=generation, species=species,
                            subspecies=subspecies)

        return messages

    def _generate_message_archive(self):
        m = stats.Messages(None)

        orig = {1: [np.zeros(shape=(350, 3)) for _ in range(300)] + [[]]}
        enc = {1: [np.zeros(shape=(350, 9)) for _ in range(300)]}
        rec = {}
        m.originals = orig
        m.encoded = enc
        m.received = rec

        return m

    def test_generationalMessages(self):
        messages = self._generate_random_messages()
        assert messages.count == 350

        messages = self._generate_ordered_messages()

        messages_101 = messages.filter([1, 0, 0])
        assert messages_101.count == 50

        messages_100_101 = messages.filter([[1, 0, 0], [1, 0, 1]])
        assert messages_100_101.count == 100

    def test_messageList(self):
        messages_list = MessageList([self._generate_ordered_messages(),
                                     self._generate_ordered_messages(run=1),
                                     self._generate_ordered_messages(generation=1),
                                     self._generate_ordered_messages(species=1),
                                     self._generate_ordered_messages(subspecies=1),
                                     self._generate_ordered_messages(run=1, generation=1),
                                     self._generate_ordered_messages(generation=1, species=1),
                                     self._generate_ordered_messages(species=1, subspecies=1),
                                     self._generate_ordered_messages(run=1, generation=1, species=1, subspecies=1),
                                     ])

        # Filter by run
        run_1 = messages_list.run(1)
        assert run_1.count == 3

        # Filter with an array
        run_01 = messages_list.run([0, 1])
        assert run_01.count == 9

        # Filter by generation
        gen_1 = messages_list.generation(1)
        assert gen_1.count == 4

        # Filter by species
        spec_1 = messages_list.species(1)
        assert spec_1.count == 4

        # Filter by subspecies
        subspec_1 = messages_list.subspecies(1)
        assert subspec_1.count == 3

        # Get encoded
        assert gen_1.encoded.shape[0] == 350 * 4
        assert gen_1.encoded.shape[1] == 9

        # Get received
        assert spec_1.received.shape[0] == 350 * 4
        assert spec_1.received.shape[1] == 9

        # Check Main List
        assert messages_list.count == 9

        # Filter by original message
        original_001 = messages_list.original([0, 0, 1])
        assert original_001.count == 9
        assert original_001.encoded.shape[0] == 50 * 9
        assert original_001.encoded.shape[1] == 9

        original_001_011 = messages_list.original([[0, 0, 1], [0, 1, 1]])
        assert original_001_011.count == 9
        assert original_001_011.encoded.shape[0] == 100 * 9
        assert original_001_011.encoded.shape[1] == 9

    def test_messageList_creation(self):
        archive = self._generate_message_archive()
        ml = MessageList.from_message_archive(archive, run=0)
        assert ml.count == 300
        np.testing.assert_equal(ml.encoded, ml.received)


class TestArchiveFitness(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.key = itertools.count()

    def _generate_sender(self, fitness=0.0, run=0, generation=0, species=0, subspecies=0,
                         fittest=False):

        k = next(self.key)
        if fittest:
            genome = DefaultGenome(key=k)
            genome.fitness = fitness
        else:
            genome = None

        return Fitness(fitness, is_sender=True, is_fittest=fittest, genome=genome,
                       run=run, generation=generation, species=species, subspecies=subspecies)

    def _generate_receiver(self, fitness=0.0, run=0, generation=0, species=0, subspecies=0,
                           fittest=False):

        k = next(self.key)
        if fittest:
            genome = DefaultGenome(key=k)
            genome.fitness = fitness
        else:
            genome = None

        return Fitness(fitness, is_sender=False, is_fittest=fittest, genome=genome,
                       run=run, generation=generation, species=species, subspecies=subspecies)

    def _generate_statistics_reporter(self):
        sr = StatisticsReporter()
        sr.generation_statistics = [
            {1: dict([(i, v) for i, v in enumerate(range(10))])},
            {1: dict([(i + 10, 2 ** v) for i, v in enumerate(range(10))])},
            {1: dict([(i + 20, 2 * v) for i, v in enumerate(range(5))]),
             2: dict([(i + 25, 4 * v) for i, v in enumerate(range(5))])},
            {1: dict([(i + 30, 100 / (v + 1)) for i, v in enumerate(range(2))]),
             2: dict([(i + 32, 4 * v + v) for i, v in enumerate(range(8))])}
        ]

        for k, v in [(9, 9), (19, 512), (29, 16), (30, 100)]:
            g = DefaultGenome(key=k)
            g.fitness = v
            sr.most_fit_genomes.append(g)

        return sr

    def test_fitness(self):
        sender = Fitness(fitness=1.0, is_sender=True)
        genome = DefaultGenome(key=10)
        genome.fitness = 2.0
        sender_fittest = Fitness(fitness=2.0, is_sender=True, is_fittest=True, genome=genome)

        assert sender.is_sender
        assert not sender.is_receiver
        assert not sender.is_fittest
        assert sender_fittest.is_sender
        assert not sender_fittest.is_receiver
        assert sender_fittest.is_fittest

        receiver = Fitness(fitness=1.0, is_sender=False)
        receiver_fittest = Fitness(fitness=2.0, is_sender=False, is_fittest=True, genome=genome)

        assert not receiver.is_sender
        assert receiver.is_receiver
        assert not receiver.is_fittest
        assert not receiver_fittest.is_sender
        assert receiver_fittest.is_receiver
        assert receiver_fittest.is_fittest

    def test_fitnessList(self):
        fitness_list = FitnessList([self._generate_sender(),
                                    self._generate_sender(fitness=1.0, fittest=True, run=1),
                                    self._generate_sender(fitness=1.0, fittest=True, generation=1),
                                    self._generate_sender(species=1),
                                    self._generate_sender(subspecies=1),
                                    self._generate_sender(run=1, generation=1),
                                    self._generate_sender(generation=1, species=1),
                                    self._generate_sender(species=1, subspecies=1),
                                    self._generate_sender(fitness=1.0, fittest=True, run=1, generation=1, species=1,
                                                          subspecies=1),
                                    self._generate_receiver(),
                                    self._generate_receiver(fitness=1.0, fittest=True, run=1),
                                    self._generate_receiver(fitness=1.0, fittest=True, generation=1),
                                    self._generate_receiver(species=1),
                                    self._generate_receiver(subspecies=1),
                                    self._generate_receiver(run=1, generation=1),
                                    self._generate_receiver(generation=1, species=1),
                                    self._generate_receiver(species=1, subspecies=1),
                                    self._generate_receiver(fitness=1.0, fittest=True, run=1, generation=1, species=1,
                                                            subspecies=1),
                                    ])

        # Filter by run
        run_1 = fitness_list.run(1)
        assert run_1.count == 6

        # Filter with an array
        run_01 = fitness_list.run([0, 1])
        assert run_01.count == 18

        # Filter by generation
        gen_1 = fitness_list.generation(1)
        assert gen_1.count == 8

        # Filter by species
        spec_1 = fitness_list.species(1)
        assert spec_1.count == 8

        # Filter by subspecies
        subspec_1 = fitness_list.subspecies(1)
        assert subspec_1.count == 6

        # Filter by fittest
        fittest_1 = fitness_list.fittest
        assert fittest_1.count == 6

        # Filter by senders
        senders_1 = fitness_list.senders
        assert senders_1.count == 9

        # Filter by receivers
        receivers_1 = fitness_list.receivers
        assert receivers_1.count == 9

    def test_fitnessList_creation(self):
        sender_sr = self._generate_statistics_reporter()
        receiver_sr = self._generate_statistics_reporter()

        fitness_list = FitnessList.from_statistics_reporters(senders=sender_sr, receivers=receiver_sr, run=0, species=0)

        # Filter by run
        run_0 = fitness_list.run(0)
        assert run_0.count == 80

        # Filter by generation
        gen_1 = fitness_list.generation(1)
        assert gen_1.count == 20

        # Filter by species
        spec_0 = fitness_list.species(0)
        assert spec_0.count == 80

        # Filter by subspecies
        subspec_1 = fitness_list.subspecies(1)
        assert subspec_1.count == 54

        # Filter by fittest
        fittest_1 = fitness_list.fittest
        assert fittest_1.count == 8

        # Filter by senders
        senders_1 = fitness_list.senders
        assert senders_1.count == 40

        # Filter by receivers
        receivers_1 = fitness_list.receivers
        assert receivers_1.count == 40

        assert fittest_1.count == len(fitness_list.generations) * 2


class TestArchiveScores(unittest.TestCase):

    def _generate_score(self, receiver=0, run=0, generation=0, species=0):
        score = Score(receiver=0, identity=[1.0 - (0.1 * i) for i in range(10)], bit=[0.1 * i for i in range(10)],
                      total=[0 for _ in range(9)] + [1],
                      run=run, generation=generation, species=species)

        return score

    def _generate_simulation_score_list(self):
        score = []
        for generation in range(10):
            d = {'species': {1: [0, 0, 0, 1, 1], 2: [1, 1, 0, 1, 1]},
                 'bit': {1: [0, 0.2, 0.4, 0.6, 0.8], 2: [1, 0.8, 0.6, 0.4, 0.2]},
                 'total': {1: [0.1, 0.2, 0.3, 0.4, 0.5], 2: [0.4, 0.5, 0.6, 0.7, 0.8]}
                 }
            score.append(d)

        return score

    def test_score(self):
        score = self._generate_score()

        assert score.count == 10
        np.testing.assert_almost_equal(score.identity, 0.55)
        np.testing.assert_almost_equal(score.bit, 0.45)
        np.testing.assert_almost_equal(score.total, 0.1)

    def test_scoreList(self):
        score_list = ScoreList.from_score_list(self._generate_simulation_score_list(), run=0, species=0)

        assert score_list.count == 20

        receiver_1 = score_list.filter(receiver=1)
        assert receiver_1.count == 10

        generation_3 = score_list.filter(generation=3)
        assert generation_3.count == 2
        np.testing.assert_almost_equal(np.average(generation_3.identity), 0.6)
        np.testing.assert_almost_equal(np.average(generation_3.bit), 0.5)
        np.testing.assert_almost_equal(np.average(generation_3.total), 0.45)

        r2_g4 = score_list.filter(generation=4, receiver=2)
        assert r2_g4.count == 1
        np.testing.assert_almost_equal(r2_g4[0].identity, 0.8)
        np.testing.assert_almost_equal(r2_g4[0].bit, 0.6)
        np.testing.assert_almost_equal(r2_g4[0].total, 0.6)


if __name__ == '__main__':
    unittest.main()
