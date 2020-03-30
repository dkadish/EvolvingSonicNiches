import logging

import numpy as np

from archive import Filterable
from archive.filterable import listify, FilterableList

logger = logging.getLogger('archive.score')


# Per Species
# decoding_scores = [ # Per Generation
#     {'species': {RECEIVER_GENOME_ID: [score1, score2, score3]}, # Called id
#      'bit': ,
#      'total': ,
#     }
# ]

class Score(Filterable):

    def __init__(self, receiver, identity: list, bit: list, total: list, run=None, generation=None, species=None,
                 subspecies=None):
        super().__init__(run, generation, species, subspecies)

        if subspecies is not None:
            logger.warning('There should not be a subspecies associated with scores.')

        self.receiver = receiver

        self._raw_identity = identity
        self._raw_bit = bit
        self._raw_total = total

    @property
    def identity(self):
        return np.average(self._raw_identity)

    @property
    def bit(self):
        return np.average(self._raw_bit)

    @property
    def total(self):
        return np.average(self._raw_total)

    @property
    def count(self):
        """Number of scores in each register, which corresponds to the number of messages received in a generation.
        This should equal the number of individuals in the population times the number of messages tested per individual.

        :return:
        """
        assert len(self._raw_identity) == len(self._raw_bit) == len(self._raw_total)

        return len(self._raw_identity)


class ScoreList(FilterableList):

    def filter(self, receiver=None, run=None, generation=None, species=None, subspecies=None):

        if subspecies is not None:
            logger.warning('There should not be a subspecies associated with scores.')

        sublist = self.data

        if receiver is not None:
            sublist = filter(lambda d: d.receiver in listify(receiver), self.data)
            return self.__class__(sublist).filter(run=run, generation=generation, species=species,
                                                  subspecies=subspecies)

        return super().filter(run=run, generation=generation, species=species, subspecies=subspecies)

    @property
    def identity(self):
        return np.array([d.identity for d in self.data])

    @property
    def bit(self):
        return np.array([d.bit for d in self.data])

    @property
    def total(self):
        return np.array([d.total for d in self.data])

    @staticmethod
    def from_score_list(scores, run: int, species: int):
        score_list = []

        for generation, d in enumerate(scores):
            identity = d['species']
            bit = d['bit']
            total = d['total']

            for r in identity:
                score = Score(receiver=r, identity=identity[r], bit=bit[r], total=total[r],
                              run=run, generation=generation, species=species)
                score_list.append(score)

        return ScoreList(score_list)
