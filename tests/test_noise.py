import unittest

import numpy as np

from noise import Noise, GenerationStepNoise, WhiteNoise


class TestNoise(unittest.TestCase):
    def test_noise(self):
        n0 = Noise(0.0, 0)
        for _ in range(10):
            np.testing.assert_equal(next(n0), np.zeros(9))

        n1 = Noise(1.0, 0)
        for _ in range(10):
            np.testing.assert_equal(next(n1), np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0]))

        n2 = Noise(2.0, 2)
        for _ in range(10):
            np.testing.assert_equal(next(n2), np.array([0, 0, 2.0, 0, 0, 0, 0, 0, 0]))

        n3 = Noise([2.0, 4.0], [2, 8])
        for _ in range(10):
            np.testing.assert_equal(next(n3), np.array([0, 0, 2.0, 0, 0, 0, 0, 0, 4.0]))


class TestGenerationNoise(unittest.TestCase):
    def test_noise(self):
        gn0 = GenerationStepNoise(0.0, 0, 0)
        for _ in range(10):
            np.testing.assert_equal(next(gn0), np.zeros(9))
            gn0.generation()

        n1 = Noise(1.0, 0)
        gn1 = GenerationStepNoise(1.0, 0, 0)
        for _ in range(10):
            np.testing.assert_equal(next(n1), next(gn1))
            gn1.generation()

        gn2 = GenerationStepNoise(2.0, 2, 2)
        for _ in range(2):
            np.testing.assert_equal(next(gn2), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
            gn2.generation()
        for _ in range(8):
            np.testing.assert_equal(next(gn2), np.array([0, 0, 2.0, 0, 0, 0, 0, 0, 0]))
            gn2.generation()

        gn3 = GenerationStepNoise([2.0, 4.0], [2, 8], 4)
        for _ in range(4):
            np.testing.assert_equal(next(gn3), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
            gn3.generation()
        for _ in range(6):
            np.testing.assert_equal(next(gn3), np.array([0, 0, 2.0, 0, 0, 0, 0, 0, 4.0]))
            gn3.generation()


class TestWhiteNoise(unittest.TestCase):

    def test_noise(self):

        # Zero noise from the first generation
        r = np.random.RandomState(0)
        wn0 = WhiteNoise(0.0, 0, random_seed=0)
        for _ in range(10):
            n = next(wn0)
            np.testing.assert_equal(n, r.random(9) * 0.0)
            np.testing.assert_equal(n, np.zeros(9))
            wn0.generation()

        # Random [0,1] noise from the first generation on channel 0
        n1 = Noise(1.0, 0)
        wn1 = WhiteNoise(1.0, 0, random_seed=0)
        r = np.random.RandomState(0)
        for _ in range(10):
            n = next(wn1)
            compare = np.zeros(9)
            compare[0] = r.random(1)
            np.testing.assert_equal(n, compare)
            wn1.generation()

        # Random [0,2.0] noise from the third generation on channel 2
        wn2 = WhiteNoise(2.0, 2, generation=2, random_seed=0)
        r = np.random.RandomState(0)
        for _ in range(2):
            n = next(wn2)
            np.testing.assert_equal(n, np.zeros(9))
            wn2.generation()
        for _ in range(8):
            n = next(wn2)
            compare = np.zeros(9)
            compare[2] = r.random(1) * 2.0
            np.testing.assert_equal(n, compare)
            wn2.generation()

        # Random [0,2.0] and [0,4.0] noise from the fifth generation on channels 2 and 8
        wn3 = WhiteNoise([2.0, 4.0], [2, 8], generation=4, random_seed=0)
        r = np.random.RandomState(0)
        for _ in range(4):
            n = next(wn3)
            np.testing.assert_equal(n, np.zeros(9))
            wn3.generation()
        for _ in range(6):
            n = next(wn3)
            compare = np.zeros(9)
            compare[[2, 8]] = r.random(2) * np.array([2.0, 4.0])
            np.testing.assert_equal(n, compare)
            wn3.generation()


if __name__ == '__main__':
    unittest.main()
