import itertools
import unittest

import numpy as np

from archive import GenerationalMessages, MessageList


def _generate_random_messages():
    original = np.random.choice([0, 1], size=(350, 3))
    encoded = np.random.random(size=(350, 9))
    noise = np.zeros(shape=(350, 9))
    noise[:, 4] += 1.0
    received = encoded + noise
    messages = GenerationalMessages(original, encoded, received, run=0, generation=0, species=0, subspecies=0)

    return messages

def _generate_ordered_messages(run=0, generation=0, species=0, subspecies=0):
    original_set = sorted(list(set(itertools.permutations([0, 0, 0, 1, 1, 1], 3))))[1:]
    original = np.concatenate([np.repeat([o], 50, axis=0) for o in original_set])
    encoded = np.average(np.mgrid[0:1:350j,0:1:9j], axis=0)
    noise = np.zeros(shape=(350, 9))
    noise[:, 4] += 1.0
    received = encoded + noise
    messages = GenerationalMessages(original, encoded, received,
                                    run=run, generation=generation, species=species, subspecies=subspecies)

    return messages

class TestArchive(unittest.TestCase):
    def test_generationalMessages(self):
        messages = _generate_random_messages()

        assert messages.count == 350

    def test_messageList(self):
        messages_list = MessageList([_generate_ordered_messages(),
                                     _generate_ordered_messages(run=1),
                                     _generate_ordered_messages(generation=1),
                                     _generate_ordered_messages(species=1),
                                     _generate_ordered_messages(subspecies=1),
                                     _generate_ordered_messages(run=1, generation=1),
                                     _generate_ordered_messages(generation=1, species=1),
                                     _generate_ordered_messages(species=1, subspecies=1),
                                     _generate_ordered_messages(run=1, generation=1, species=1, subspecies=1),
                                     ])

        # Filter by run
        run_1 = messages_list.run(1)
        assert run_1.count() == 3

        # Filter with an array
        run_01 = messages_list.run([0, 1])
        assert run_01.count() == 9

        # Filter by generation
        gen_1 = messages_list.generation(1)
        assert gen_1.count() == 4

        # Filter by species
        spec_1 = messages_list.species(1)
        assert spec_1.count() == 4

        # Filter by subspecies
        subspec_1 = messages_list.subspecies(1)
        assert subspec_1.count() == 3

        # # Get originals
        # assert run_1.original.shape[0] == 350*3
        # assert run_1.original.shape[1] == 3

        # Get encoded
        assert gen_1.encoded.shape[0] == 350*4
        assert gen_1.encoded.shape[1] == 9

        # Get received
        assert spec_1.received.shape[0] == 350*4
        assert spec_1.received.shape[1] == 9

        # Check Main List
        assert messages_list.count() == 9

        # Filter by original message
        original_001 = messages_list.original([0, 0, 1])
        assert original_001.count() == 9
        assert original_001.encoded.shape[0] == 50*9
        assert original_001.encoded.shape[1] == 9

        original_001_011 = messages_list.original([[0, 0, 1], [0, 1, 1]])
        assert original_001_011.count() == 9
        assert original_001_011.encoded.shape[0] == 100*9
        assert original_001_011.encoded.shape[1] == 9

# class TestGenerationNoise(unittest.TestCase):
#     def test_noise(self):
#         gn0 = GenerationStepNoise(0.0, 0, 0)
#         for _ in range(10):
#             np.testing.assert_equal(next(gn0), np.zeros(9))
#             gn0.generation()
#
#         n1 = Noise(1.0, 0)
#         gn1 = GenerationStepNoise(1.0, 0, 0)
#         for _ in range(10):
#             np.testing.assert_equal(next(n1), next(gn1))
#             gn1.generation()
#
#         gn2 = GenerationStepNoise(2.0, 2, 2)
#         for _ in range(2):
#             np.testing.assert_equal(next(gn2), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
#             gn2.generation()
#         for _ in range(8):
#             np.testing.assert_equal(next(gn2), np.array([0, 0, 2.0, 0, 0, 0, 0, 0, 0]))
#             gn2.generation()
#
#         gn3 = GenerationStepNoise([2.0, 4.0], [2, 8], 4)
#         for _ in range(4):
#             np.testing.assert_equal(next(gn3), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
#             gn3.generation()
#         for _ in range(6):
#             np.testing.assert_equal(next(gn3), np.array([0, 0, 2.0, 0, 0, 0, 0, 0, 4.0]))
#             gn3.generation()
#
#
# class TestWhiteNoise(unittest.TestCase):
#
#     def test_noise(self):
#
#         # Zero noise from the first generation
#         r = np.random.RandomState(0)
#         wn0 = WhiteNoise(0.0, 0, random_seed=0)
#         for _ in range(10):
#             n = next(wn0)
#             np.testing.assert_equal(n, r.random(9) * 0.0)
#             np.testing.assert_equal(n, np.zeros(9))
#             wn0.generation()
#
#         # Random [0,1] noise from the first generation on channel 0
#         n1 = Noise(1.0, 0)
#         wn1 = WhiteNoise(1.0, 0, random_seed=0)
#         r = np.random.RandomState(0)
#         for _ in range(10):
#             n = next(wn1)
#             compare = np.zeros(9)
#             compare[0] = r.random(1)
#             np.testing.assert_equal(n, compare)
#             wn1.generation()
#
#         # Random [0,2.0] noise from the third generation on channel 2
#         wn2 = WhiteNoise(2.0, 2, generation=2, random_seed=0)
#         r = np.random.RandomState(0)
#         for _ in range(2):
#             n = next(wn2)
#             np.testing.assert_equal(n, np.zeros(9))
#             wn2.generation()
#         for _ in range(8):
#             n = next(wn2)
#             compare = np.zeros(9)
#             compare[2] = r.random(1) * 2.0
#             np.testing.assert_equal(n, compare)
#             wn2.generation()
#
#         # Random [0,2.0] and [0,4.0] noise from the fifth generation on channels 2 and 8
#         wn3 = WhiteNoise([2.0, 4.0], [2, 8], generation=4, random_seed=0)
#         r = np.random.RandomState(0)
#         for _ in range(4):
#             n = next(wn3)
#             np.testing.assert_equal(n, np.zeros(9))
#             wn3.generation()
#         for _ in range(6):
#             n = next(wn3)
#             compare = np.zeros(9)
#             compare[[2, 8]] = r.random(2) * np.array([2.0, 4.0])
#             np.testing.assert_equal(n, compare)
#             wn3.generation()


if __name__ == '__main__':
    unittest.main()
