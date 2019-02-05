from multiprocessing import Queue

import numpy as np
from scipy.spatial.distance import cdist

from messaging import MessageType, Message


class EncodedStatsBase:
    def __init__(self, messages: Queue):
        self.messages = messages

    def run(self):
        message = self.messages.get()
        while message.type != MessageType.FINISHED:
            self.handle_message(message)

            message = self.messages.get()
            while message.type == MessageType.GENERATION:
                self.handle_generation(message)
                message = self.messages.get()

        self.handle_finish()

    def handle_message(self, message):
        pass

    def handle_generation(self, message):
        pass

    def handle_finish(self):
        pass

    def done(self):
        self.messages.put(Message.Finished())

class Spectrum(EncodedStatsBase):

    def __init__(self, messages: Queue):
        super(Spectrum, self).__init__(messages)

        self.spectra = {}
        self.n_spectra = None

    def handle_message(self, message):
        super(Spectrum, self).handle_message(message)

        species = message.species_id
        encoded_message = message.message['encoded']

        if self.n_spectra is None:
            self.n_spectra = len(encoded_message)

        if species not in self.spectra:
            self.spectra[species] = [np.zeros(self.n_spectra)]

        self.spectra[species][-1] += np.array(encoded_message)

    def handle_generation(self, message):
        super(Spectrum, self).handle_generation(message)

        species = message.species_id
        self.spectra[species].append(np.zeros(self.n_spectra))


class Loudness(EncodedStatsBase):

    def __init__(self, messages: Queue):
        super(Loudness, self).__init__(messages)

        self.loudness = {}
        self.avg = {}
        self.std = {}

    def handle_message(self, message):
        '''Add message to list for generation

        :param Message message: The message to process
        :return: None
        '''
        super(Loudness, self).handle_message(message)

        species = message.species_id
        encoded_message = message.message['encoded']

        if species not in self.loudness:
            self.loudness[species] = []

        self.loudness[species].extend(encoded_message)

    def handle_generation(self, message):
        super(Loudness, self).handle_generation(message)

        species = message.species_id

        if species not in self.avg:
            self.avg[species] = []

        if species not in self.std:
            self.std[species] = []

        l_avg = np.average(self.loudness[species])
        l_std = np.std(self.loudness[species])

        self.avg[species].append(l_avg)
        self.std[species].append(l_std)
        self.loudness[species] = []


class Cohesion(EncodedStatsBase):

    def __init__(self, messages:Queue):
        super(Cohesion, self).__init__(messages)

        self.cohesion = {}
        self.avg = {}
        self.std = {}
        
    def handle_message(self, message: Message):
        '''Handles individual messages for the Cohesion Statistics module.

        Adds the incoming message to a dict consisting of ``{ original message: [encoded message, ] }``.
        This structure is reset every generation.

        :param messaging.Message message: The message to process.
        :return: None
        '''
        super(Cohesion, self).handle_message(message)

        species = message.species_id
        if species not in self.cohesion:
            self.cohesion[species] = {}

        original = message.message['original']
        encoded = message.message['encoded']
        if original not in self.cohesion[species]:
            self.cohesion[species][original] = []
        self.cohesion[species][original].append(encoded)

    def handle_generation(self, message):
        '''Handles the end of a generation for cohesion stats.

        Records the avg and std of the distances between encodings of the same message within a species and generation.
        Finds the distance matrix for all encoded messages of the same species/generation/original.

        :todo Should this be weighted?:

        :param message:
        :return:
        '''
        super(Cohesion, self).handle_generation(message)

        species = message.species_id

        if species not in self.avg:
            self.avg[species] = []

        if species not in self.std:
            self.std[species] = []

        distances = []
        for m in self.cohesion[species]: # For the same original message
            encodings = np.array(self.cohesion[species][m])
            distance_matrix = cdist(encodings, encodings)
            distances.append(np.average(distance_matrix))

        self.avg[species].append(np.average(distances))
        self.std[species].append(np.std(distances))
        self.cohesion[species] = {}
