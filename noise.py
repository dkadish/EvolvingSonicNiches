from collections import Iterable

import numpy as np

N_CHANNELS = 9


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

    def __init__(self, levels, channels, generation) -> None:
        super().__init__(levels, channels)

        self.current_generation = 0
        self.start_generation = generation

    def __next__(self):

        if self.current_generation >= self.start_generation:
            return super().__next__()
        else:
            return np.zeros(N_CHANNELS)

    def generation(self):
        self.current_generation += 1


if __name__ == '__main__':
    n = Noise(1.0, 0)

    print(next(n))

    n = Noise([1.0, 2.0, 3.0], [0, 3, 6])

    print(next(n))
