import numpy as np

from archive import Filterable
from archive.filterable import FilterableList


class Messages(Filterable):
    """Holds an ordered set of messages from a generation of simulation.

    The messages are stored in 3 N-length arrays, where N is the number of messages sent in the generation.
    The three arrays, original, encoded, and received represent the various stages of the message in simulation.

    """

    def __init__(self, original, encoded, received, run=None, generation=None, species=None, subspecies=None):
        super().__init__(run, generation, species, subspecies)

        self.original = original
        self.encoded = encoded
        self.received = received

        self._original_packed = np.packbits(np.bool8(self.original), axis=1)

    @property
    def count(self):
        """Number of messages transmitted in the set.

        :return:
        """
        assert self.original.shape[0] == self.encoded.shape[0] == self.received.shape[0]
        return self.original.shape[0]

    @property
    def channels(self):
        """Number of channels for messaging.

        :return:
        """
        assert self.encoded.shape[1] == self.received.shape[1]
        return self.encoded.shape[1]

    def filter(self, original):
        """Filter by original message.

        :param original:
        :return: A new instance of GenerationalMessage with only the messages that originated as original.
        """
        try:
            packed = np.packbits(original, axis=1)
        except np.AxisError:
            packed = np.packbits([original], axis=1)

        mask = np.isin(self._original_packed, packed).flatten()

        return self.__class__(self.original[mask], self.encoded[mask], self.received[mask],
                              run=self.run, generation=self.generation,
                              species=self.species, subspecies=self.subspecies)


class MessageList(FilterableList):

    def filter(self, run=None, generation=None, species=None, subspecies=None, original=None):

        if original is not None:
            sublist = self.data
            sublist = [d.filter(original) for d in self.data]
            return self.__class__(sublist).filter(run=run, generation=generation, species=species, subspecies=subspecies)

        return super().filter(run=run, generation=generation, species=species, subspecies=subspecies)

    def original(self, o):
        return self.filter(original=o)

    @property
    def encoded(self):
        return np.concatenate([d.encoded for d in self.data], axis=0)

    @property
    def received(self):
        return np.concatenate([d.received for d in self.data], axis=0)

    @staticmethod
    def from_message_archive(message_archive, run):
        species = list(message_archive.originals.keys())
        ml = MessageList()
        for s in species:
            for g in range(len(message_archive.originals[s])):
                if type(message_archive.originals[s][g]) == list and len(message_archive.originals[s][g]) == 0:
                    break
                original = message_archive.originals[s][g]
                encoded = message_archive.encoded[s][g]
                try:
                    received = message_archive.received[s][g]
                except KeyError:
                    received = message_archive.encoded[s][g]
                m = Messages(original, encoded, received, run=run, generation=g, species=s)

                ml.append(m)

        return ml