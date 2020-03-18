from multiprocessing import Queue

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.manifold import TSNE

from messaging import MessageType, Message


class EncodedStatsBase:
    def __init__(self, messages: Queue):
        self.messages = messages
        self.species = set()

        self._species_generation_complete = set()

    def run(self):
        message = self.messages.get()
        while message.type != MessageType.FINISHED:
            self.species.add(message.species_id)
            self.handle_message(message)

            message = self.messages.get()
            while message.type == MessageType.GENERATION:
                self.handle_generation(message)
                self._species_generation_complete.add(message.species_id)

                # All species have completed a generation
                if self._species_generation_complete == self.species:
                    self.handle_full_generation(message)

                    self._species_generation_complete = set()

                message = self.messages.get()

        self.handle_finish()

    def handle_message(self, message):
        pass

    def handle_generation(self, message):
        pass

    def handle_full_generation(self, message):
        pass

    def handle_finish(self):
        pass

    def done(self):
        self.messages.put(Message.Finished())

    @property
    def n_species(self):
        return len(self.species)


class Spectrum(EncodedStatsBase):
    """Spectrum for all messages sent by a given species.

    For each generation, aggregates messages sent by a species to allow for the production of a generational
    spectrogram.
    """

    def __init__(self, messages: Queue):
        super(Spectrum, self).__init__(messages)

        self.spectra = {}
        self.received_spectra = {}
        self.n_spectra = None

    def handle_message(self, message):
        super(Spectrum, self).handle_message(message)

        species = message.species_id
        encoded_message = message.message['encoded']
        received_message = message.message['received']

        # Figure out the number of channels, if not set
        if self.n_spectra is None:
            self.n_spectra = len(encoded_message)

        if species not in self.spectra:
            self.spectra[species] = [np.zeros(self.n_spectra)]
        if species not in self.received_spectra:
            self.received_spectra[species] = [np.zeros(self.n_spectra)]

        self.spectra[species][-1] += np.array(encoded_message)
        self.received_spectra[species][-1] += np.array(received_message)

    def handle_generation(self, message):
        super(Spectrum, self).handle_generation(message)

        species = message.species_id
        self.spectra[species].append(np.zeros(self.n_spectra))
        self.received_spectra[species].append(np.zeros(self.n_spectra))


class MessageSpectrum(EncodedStatsBase):
    """Spectra for each type of message sent by every species.

    For each generation, aggregates messages sent by a species divided by the original message
    to allow for the production of a generational spectrogram for each message ([0,0,1]...[1,1,1]).
    """

    def __init__(self, messages: Queue):
        super(MessageSpectrum, self).__init__(messages)

        self.spectra = {}
        self.received_spectra = {}
        self.n_spectra = None

    def handle_message(self, message):
        super(MessageSpectrum, self).handle_message(message)

        species = message.species_id
        original_message = message.message['original']
        encoded_message = message.message['encoded']
        received_message = message.message['received']

        if self.n_spectra is None:
            self.n_spectra = len(encoded_message)

        if species not in self.spectra:
            self.spectra[species] = {}
        if species not in self.received_spectra:
            self.received_spectra[species] = {}

        if original_message not in self.spectra[species]:
            self.spectra[species][original_message] = [np.zeros(self.n_spectra)]
        if original_message not in self.received_spectra[species]:
            self.received_spectra[species][original_message] = [np.zeros(self.n_spectra)]

        self.spectra[species][original_message][-1] += np.array(encoded_message)
        self.received_spectra[species][original_message][-1] += np.array(received_message)

    def handle_generation(self, message):
        super(MessageSpectrum, self).handle_generation(message)

        species = message.species_id
        for m in self.spectra[species]:
            self.spectra[species][m].append(np.zeros(self.n_spectra))
        for m in self.received_spectra[species]:
            self.received_spectra[species][m].append(np.zeros(self.n_spectra))


class Loudness(EncodedStatsBase):

    def __init__(self, messages: Queue):
        super(Loudness, self).__init__(messages)

        self.loudness = {}
        self.avg = {}
        self.std = {}

    def handle_message(self, message):
        """Add message to list for generation

        :param Message message: The message to process
        :return: None
        """
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

    def __init__(self, messages: Queue):
        super(Cohesion, self).__init__(messages)

        self.cohesion = {}
        self.avg = {}
        self.std = {}

    def handle_message(self, message: Message):
        """Handles individual messages for the Cohesion Statistics module.

        Adds the incoming message to a dict consisting of ``{ original message: [encoded message, ] }``.
        This structure is reset every generation.

        :param messaging.Message message: The message to process.
        :return: None
        """
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
        """Handles the end of a generation for cohesion stats.

        Records the avg and std of the distances between encodings of the same message within a species and generation.
        Finds the distance matrix for all encoded messages of the same species/generation/original.

        :todo Should this be weighted?:

        :param message:
        :return:
        """
        super(Cohesion, self).handle_generation(message)

        species = message.species_id

        if species not in self.avg:
            self.avg[species] = []

        if species not in self.std:
            self.std[species] = []

        distances = []
        for m in self.cohesion[species]:  # For the same original message
            encodings = np.array(self.cohesion[species][m])
            distance_matrix = cdist(encodings, encodings)
            distances.append(np.average(distance_matrix))

        self.avg[species].append(np.average(distances))
        self.std[species].append(np.std(distances))
        self.cohesion[species] = {}


class MessageArchive:
    def __init__(self, generation: int, name: str, messages: np.array, labels: np.array):
        self.generation = generation
        self.name = name
        self.messages = messages
        self.labels = labels
        self._2d = None

    @property
    def two_dimensional(self):
        if self._2d is None:
            self._2d = TSNE(n_components=2).fit_transform(self.messages)

        return self._2d

    @property
    def numeric_labels(self):
        if type(self.name) == int:
            # These are triplet values
            return np.sum(np.array(self.labels) << np.arange(2, -1, -1), axis=1)

        else:
            return [int(l) for l in self.labels]


class Cluster(EncodedStatsBase):
    overall = 'ALL'
    archive_interval = 50  # 50

    def __init__(self, messages: Queue):
        super(Cluster, self).__init__(messages)

        self.encoded = {}
        self.originals = {}

        self.ch = {}
        self.silhouette = {}

        self.archive = {}

        self.best = {}

    @property
    def generation(self):
        return min([len(v) for v in self.ch.values()])

    def handle_message(self, message: Message):
        """Handles individual messages for the Cohesion Statistics module.

        Adds the incoming message to a dict consisting of ``{ original message: [encoded message, ] }``.
        This structure is reset every generation.

        :param messaging.Message message: The message to process.
        :return: None
        """
        super(Cluster, self).handle_message(message)

        species = message.species_id
        if species not in self.encoded:
            self.encoded[species] = []
        if species not in self.originals:
            self.originals[species] = []

        original = message.message['original']
        encoded = message.message['encoded']
        self.originals[species].append(original)
        self.encoded[species].append(encoded)

    def handle_generation(self, message):
        super(Cluster, self).handle_generation(message)

        species = message.species_id
        messages = np.array(self.encoded[species])
        originals = np.array(self.originals[species])
        # Convert to integer
        original_ints = np.sum(originals << np.array([2, 1, 0]), axis=1)

        if species not in self.ch:
            self.ch[species] = []
        if species not in self.silhouette:
            self.silhouette[species] = []
            self.silhouette['{}.{}'.format(species, 0)] = []
            self.silhouette['{}.{}'.format(species, 1)] = []
            self.silhouette['{}.{}'.format(species, 2)] = []
        if species not in self.archive:
            self.archive[species] = []
        if species not in self.best:
            self.best[species] = 0

        # Calculate within species scores for this generation
        self.ch[species].append(metrics.calinski_harabaz_score(messages, original_ints))
        self.silhouette[species].append(metrics.silhouette_score(messages, original_ints))

        # Calculate scores for each bit
        for b in range(3):
            s = metrics.silhouette_score(messages, originals[:, 2 - b])
            self.silhouette['{}.{}'.format(species, b)].append(s)

        # log best score
        is_best = False
        best_multiplier = 1.15
        print(species, self.silhouette[species][-1] + 1, self.best[species] * best_multiplier,
              (self.silhouette[species][-1] + 1) > self.best[species] * best_multiplier)
        if (self.silhouette[species][-1] + 1) > self.best[
                species] * best_multiplier:  # +1 to raise range to [0,2] from [-1,1]
            is_best = True
            self.best[species] = self.silhouette[species][-1] + 1

        if self.overall not in self.encoded:
            self.encoded[self.overall] = []
        if self.overall not in self.originals:
            self.originals[self.overall] = []

        # Add to overall lists for between species comparison
        self.encoded[self.overall].extend(self.encoded[species])
        self.originals[self.overall].extend([species for o in self.originals[species]])

        if self.generation == 1 or self.generation % self.archive_interval == 0 or is_best:
            self.archive[species].append(
                MessageArchive(self.generation, species, self.encoded[species], self.originals[species]))

        # Delete messages from this generation
        self.encoded[species] = []
        self.originals[species] = []

    def handle_full_generation(self, message):
        super(Cluster, self).handle_full_generation(message)

        messages = np.array(self.encoded[self.overall])
        originals = np.array(self.originals[self.overall])

        if self.overall not in self.ch:
            self.ch[self.overall] = []
        if self.overall not in self.silhouette:
            self.silhouette[self.overall] = []
        if self.overall not in self.archive:
            self.archive[self.overall] = []
        if self.overall not in self.best:
            self.best[self.overall] = 0

        try:
            self.ch[self.overall].append(metrics.calinski_harabaz_score(messages, originals))
            self.silhouette[self.overall].append(metrics.silhouette_score(messages, originals))
        except ValueError as e:
            print('ValueError. Likely only one species available. Ignoring clustering scores.')

        # log best score
        is_best = False
        print(self.overall, self.silhouette[self.overall][-1] + 1, self.best[self.overall] * 1.2,
              (self.silhouette[self.overall][-1] + 1) > self.best[self.overall] * 1.2)
        if (self.silhouette[self.overall][-1] + 1) > self.best[
                self.overall] * 1.2:  # +1 to raise range to [0,2] from [-1,1]
            is_best = True
            self.best[self.overall] = self.silhouette[self.overall][-1] + 1

        if self.generation % self.archive_interval == 0 or is_best:
            self.archive[self.overall].append(
                MessageArchive(self.generation, self.overall, self.encoded[self.overall], self.originals[self.overall]))

        # Delete messages from this generation
        self.encoded[self.overall] = []
        self.originals[self.overall] = []


class Messages(EncodedStatsBase):

    def __init__(self, messages: Queue):
        super(Messages, self).__init__(messages)

        self.encoded = {} # Message as encoded by the senders.
        self.originals = {} # This is the original message that was sent
        self.received = {} # The message as received by the receiver. Includes noise, interference, distortion, etc.

    def handle_message(self, message: Message):
        super(Messages, self).handle_message(message)

        species = message.species_id
        if species not in self.encoded:
            self.encoded[species] = [[]]
        if species not in self.originals:
            self.originals[species] = [[]]
        if species not in self.received:
            self.received[species] = [[]]

        original = message.message['original']
        encoded = message.message['encoded']
        received = message.message['received']
        self.originals[species][-1].append(original)
        self.encoded[species][-1].append(encoded)
        self.received[species][-1].append(received)

    def handle_generation(self, message):
        super(Messages, self).handle_generation(message)

        species = message.species_id

        # Reformat the last generation as a numpy array
        self.encoded[species][-1] = np.array(self.encoded[species][-1])
        self.originals[species][-1] = np.array(self.originals[species][-1])
        self.received[species][-1] = np.array(self.received[species][-1])

        # Append a new, empty array
        self.encoded[species].append([])
        self.originals[species].append([])
        self.received[species].append([])

    def handle_finish(self):
        super(Messages, self).handle_finish()

        for s in self.encoded:
            if len(self.encoded[s][-1]) == 0:
                del self.encoded[s][-1]

        for s in self.received:
            if len(self.received[s][-1]) == 0:
                del self.received[s][-1]
