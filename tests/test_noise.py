import unittest

import numpy as np

from noise import Noise, GenerationStepNoise


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


if __name__ == '__main__':
    unittest.main()
