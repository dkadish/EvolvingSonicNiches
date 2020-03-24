import itertools
import unittest

import numpy as np

from archive import GenerationalMessages, MessageList
from stats import Messages


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

def _generate_message_archive():
    m = Messages(None)

    orig = {1: [np.zeros(shape=(350, 3)) for _ in range(300)] + [[]]}
    enc = {1: [np.zeros(shape=(350, 9)) for _ in range(300)]}
    rec = {}
    m.originals = orig
    m.encoded = enc
    m.received = rec

    return m

class TestArchive(unittest.TestCase):
    def test_generationalMessages(self):
        messages = _generate_random_messages()
        assert messages.count == 350

        messages = _generate_ordered_messages()

        messages_101 = messages.filter([1, 0, 0])
        assert messages_101.count == 50

        messages_100_101 = messages.filter([[1, 0, 0], [1, 0, 1]])
        assert messages_100_101.count == 100

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
        assert messages_list.count == 9

        # Filter by original message
        original_001 = messages_list.original([0, 0, 1])
        assert original_001.count == 9
        assert original_001.encoded.shape[0] == 50*9
        assert original_001.encoded.shape[1] == 9

        original_001_011 = messages_list.original([[0, 0, 1], [0, 1, 1]])
        assert original_001_011.count == 9
        assert original_001_011.encoded.shape[0] == 100*9
        assert original_001_011.encoded.shape[1] == 9

    def test_messageList_creation(self):
        archive = _generate_message_archive()
        ml = MessageList.from_message_archive(archive)
        print(ml.count)
        assert ml.count == 300
        np.testing.assert_equal(ml.encoded, ml.received)

if __name__ == '__main__':
    unittest.main()
