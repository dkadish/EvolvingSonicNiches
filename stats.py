from multiprocessing import Queue

import numpy as np

from messaging import MessageType


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

    def handle_message(self, message):
        pass

    def handle_generation(self, message):
        pass

    def handle_finish(self):
        pass

    def done(self):
        self.messages.put(False)


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

    def handle_finish(self):
        pass


class Loudness(EncodedStatsBase):
    pass


class Cohesion(EncodedStatsBase):
    pass
