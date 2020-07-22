import logging
from collections import Iterable

import numpy as np

N_CHANNELS = 9

logger = logging.getLogger('noise')


class Noise(Iterable):
    """Outputs a constant noise for across all generations and individuals."""

    def __init__(self, levels, channels) -> None:
        super().__init__()
        self.channels = channels
        self.levels = levels

    def __iter__(self):
        return self

    def __next__(self):
        noises = np.zeros(N_CHANNELS)
        noises[self.channels] = self.levels
        return noises

    def generation(self):
        pass


class GenerationStepNoise(Noise):
    """Noise that appears suddenly one generation."""

    def __init__(self, levels, channels, generation, current_generation=0) -> None:
        super().__init__(levels, channels)

        self.current_generation = current_generation
        self.start_generation = generation

    def __next__(self):

        if self.current_generation >= self.start_generation:
            return super().__next__()
        else:
            return np.zeros(N_CHANNELS)

    def generation(self):
        self.current_generation += 1


class WhiteNoise(GenerationStepNoise):
    """Noise that appears suddenly one generation."""

    def __init__(self, levels, channels, generation: int = 0, random_seed: int = None) -> None:
        """

        :param levels: Levels sets the max value for the noise, which ranges from [0,level]
        :param channels:
        :param generation:
        :param random_seed:
        """
        super().__init__(levels, channels, generation)

        self.current_generation = 0
        self.start_generation = generation

        self.random = np.random
        if random_seed is not None:
            logging.debug('Setting random seed to {}'.format(random_seed))
            self.random = np.random.RandomState(random_seed)

        self.n_channels = type(channels) == int and 1 or len(channels)

    def __next__(self):

        if self.current_generation >= self.start_generation:
            r = self.random.random(self.n_channels) * np.array(self.levels)
            noise = np.zeros(N_CHANNELS)
            noise[self.channels] += r
            return noise
        else:
            return np.zeros(N_CHANNELS)

    def generation(self):
        self.current_generation += 1


if __name__ == '__main__':
    n = Noise(1.0, 0)

    print(next(n))

    n = Noise([1.0, 2.0, 3.0], [0, 3, 6])

    print(next(n))
