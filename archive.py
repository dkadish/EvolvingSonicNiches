from collections import UserDict, UserList
import numpy as np

class Filterable:

    def __init__(self, run=None, generation=None, species=None, subspecies=None):

        self.run = run
        self.generation = generation
        self.species = species
        self.subspecies = subspecies

class GenerationalMessages(Filterable):
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

class Spectrum(Filterable):

    def __init__(self, run=None, generation=None, species=None, subspecies=None):
        super().__init__(run, generation, species, subspecies)

class Stats(Filterable):
    pass

class FilterableList(UserList):

    def generation(self, g):
        """Returns a FilterableList with only items in the specified generation(s)

        :return: FilterableList with items from generation g
        """
        return self.filter(generation=g)

    def species(self, s):
        """Returns a FilterableList with only items in the specified species.

        Species here refers to experimental species (for example, if there are two distinct
        species communicating within a soundscape).

        :return: FilterableList with items from species s
        """
        return self.filter(species=s)

    def subspecies(self, s):
        """Returns a FilterableList with only items in the specified subspecies

        Subspecies refers to the NEAT concept of species, in which an evolutionary population
        is subdivided to maintain diversity. Here, NEAT species are called subspecies.

        :return: FilterableList with items from subspecies s
        """
        return self.filter(subspecies=s)

    def run(self, r):
        """Returns a FilterableList with only items from the specified run(s)

        :return: FilterableList with items from run r
        """
        return self.filter(run=r)

    def filter(self, run=None, generation=None, species=None, subspecies=None):
        """A general filter for paring the list down by multiple parameters

        :param run: int or list of runs to select
        :param generation: int or list of generations to select
        :param species: int or list of species to select
        :param subspecies: int or list of subspecies to select
        :return:
        """
        sublist = self.data

        if run is not None:
            sublist = filter(lambda i: i.run in listify(run), sublist)

        if generation is not None:
            sublist = filter(lambda i: i.generation in listify(generation), sublist)

        if species is not None:
            sublist = filter(lambda i: i.species in listify(species), sublist)

        if subspecies is not None:
            sublist = filter(lambda i: i.subspecies in listify(subspecies), sublist)

        return self.__class__(sublist)

    @property
    def count(self):
        return len(self.data)

class MessageList(FilterableList):

    def filter(self, run=None, generation=None, species=None, subspecies=None, original=None):

        if original is not None:
            sublist = self.data
            sublist = [d.filter(original) for d in self.data]
            return self.__class__(sublist).filter(run, generation, species, subspecies)

        return super().filter(run, generation, species, subspecies)

    # @property
    # def original(self):
    #     return np.concatenate([d.original for d in self.data], axis=0)

    def original(self, o):
        return self.filter(original=o)

    @property
    def encoded(self):
        return np.concatenate([d.encoded for d in self.data], axis=0)

    @property
    def received(self):
        return np.concatenate([d.received for d in self.data], axis=0)

    @staticmethod
    def from_message_archive(message_archive, run=None):
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
                m = GenerationalMessages(original, encoded, received, run=run, generation=g, species=s)

                ml.append(m)

        return ml

def listify(v):
    if type(v) is int:
        return [v]
    return v

class Archive:

    def __init__(self, messages=None, stats={}, configs={}, **kwargs) -> None:
        self.messages = messages

    def add_run(self, message_archive):
        pass

    @staticmethod
    def createArchive(message_archive):
        pass

class Archive(UserDict):

    def __init__(self, messages={}, stats={}, configs={}, **kwargs) -> None:
        super().__init__({}, **kwargs)

        self['messages'] = self.Messages(**messages)
        self['stats'] = self.Stats(**stats)
        self['configs'] = self.Configs(**configs)

    class Messages(UserDict):

        def __init__(self, original={}, encoded={}, received={}, **kwargs) -> None:
            super().__init__(dict, **kwargs)

            self['original'] = original
            self['encoded'] = encoded
            self['received'] = received

        @property
        def original(self):
            return self['original']

        @property
        def encoded(self):
            return self['encoded']

        @property
        def received(self):
            return self['received']

    class Stats(UserDict):


        @property
        def encoder(self):
            pass

        @property
        def decoder(self):
            pass

    class Configs(UserDict):
        pass

    @staticmethod
    def createArchive(config, evaluator_config, encoder_stats, decoder_stats,
                  message_spectra, received_message_spectra, scores,
                  generations, noise_channel, noise_level, messages):
        pass