from unittest import TestCase

from evolvingniches.evaluators import DecoderEvaluator


class TestDecoderEvaluator(TestCase):

    def test_sender_fitness(self):
        # Everything correct
        self.assertEqual(DecoderEvaluator.sender_fitness([0, 0, 0], [0, 0, 0, 1], 0, 0), 4.0)

        # One bit incorrect
        self.assertEqual(DecoderEvaluator.sender_fitness([0, 0, 0], [1, 0, 0, 1], 0, 0), 3.0)

        # Two bits incorrect
        self.assertEqual(DecoderEvaluator.sender_fitness([0, 0, 1], [0, 1, 0, 1], 0, 0), 2.0)

        # Three bits incorrect
        self.assertEqual(DecoderEvaluator.sender_fitness([0, 1, 0], [1, 0, 1, 1], 0, 0), 1.0)

        # Bits correct, wrong species
        self.assertEqual(DecoderEvaluator.sender_fitness([1, 0, 0], [1, 0, 0, 1], 0, 1), None)

        # One bit incorrect
        self.assertEqual(DecoderEvaluator.sender_fitness([0, 0, 0], [0, 1, 0, 1], 1, 0), None)

        # Bits correct, wrong species identification
        self.assertEqual(DecoderEvaluator.sender_fitness([1, 0, 0], [1, 0, 0, 0], 1, 1), 0.0)

        # One bit incorrect, wrong species identification
        self.assertEqual(DecoderEvaluator.sender_fitness([0, 0, 0], [0, 1, 0, 0], 0, 0), 0.0)

    def test_receiver_fitness(self):
        # Everything correct
        self.assertEqual(DecoderEvaluator.receiver_fitness([0, 0, 0], [0, 0, 0, 1], 0, 0), 4.0)

        # One bit incorrect
        self.assertEqual(DecoderEvaluator.receiver_fitness([0, 0, 0], [1, 0, 0, 1], 0, 0), 3.0)

        # Two bits incorrect
        self.assertEqual(DecoderEvaluator.receiver_fitness([0, 0, 1], [0, 1, 0, 1], 0, 0), 2.0)

        # Three bits incorrect
        self.assertEqual(DecoderEvaluator.receiver_fitness([0, 1, 0], [1, 0, 1, 1], 0, 0), 1.0)

        # Bits correct, wrong species
        self.assertEqual(DecoderEvaluator.receiver_fitness([1, 0, 0], [1, 0, 0, 1], 0, 1), 0.0)

        # One bit incorrect
        self.assertEqual(DecoderEvaluator.receiver_fitness([0, 0, 0], [0, 1, 0, 1], 1, 0), 0.0)

        # Bits correct, wrong species identification
        self.assertEqual(DecoderEvaluator.receiver_fitness([1, 0, 0], [1, 0, 0, 0], 1, 1), 0.0)

        # One bit incorrect, wrong species identification
        self.assertEqual(DecoderEvaluator.receiver_fitness([0, 0, 0], [0, 1, 0, 0], 0, 0), 0.0)

    def test_score(self):
        # Everything correct
        self.assertEqual(DecoderEvaluator.score([0, 0, 0], [0, 0, 0, 1], 0, 0), [1.0, 1.0, 1.0])

        # One bit incorrect
        self.assertEqual(DecoderEvaluator.score([0, 0, 0], [1, 0, 0, 1], 0, 0), [1.0, 2.0 / 3.0, 0.0])

        # Two bits incorrect
        self.assertEqual(DecoderEvaluator.score([0, 0, 1], [0, 1, 0, 1], 0, 0), [1.0, 1.0 / 3.0, 0.0])

        # Three bits incorrect
        self.assertEqual(DecoderEvaluator.score([0, 1, 0], [1, 0, 1, 1], 0, 0), [1.0, 0.0, 0.0])

        # Bits correct, wrong species
        self.assertEqual(DecoderEvaluator.score([1, 0, 0], [1, 0, 0, 1], 0, 1), [0.0, None, None])

        # One bit incorrect
        self.assertEqual(DecoderEvaluator.score([0, 0, 0], [0, 1, 0, 1], 1, 0), [0.0, None, None])

        # Bits correct, wrong species identification
        self.assertEqual(DecoderEvaluator.score([1, 0, 0], [1, 0, 0, 0], 1, 1), [0.0, 0.0, 0.0])

        # One bit incorrect, wrong species identification
        self.assertEqual(DecoderEvaluator.score([0, 0, 0], [0, 1, 0, 0], 0, 0), [0.0, 0.0, 0.0])
