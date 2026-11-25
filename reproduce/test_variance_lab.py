import math
import unittest

from variance_lab import CASES, coefficients, compare, gaussian_pairs, sign_cards, variance


class VarianceTests(unittest.TestCase):
    def test_exact_population_limits_and_covariance(self):
        for share in (0., .17, .5, .64, 1.):
            for mode in CASES:
                rows = compare(share, mode, [(0., 1.), (1., -1.)])
                for row, (a, b) in zip(rows, coefficients(share)):
                    expected = {"independent": a*a+b*b, "fixed": b*b, "shared": (a+b)**2}
                    with self.subTest(share=share, mode=mode, a=a):
                        self.assertAlmostEqual(row["mean"], .8*a if mode=="fixed" else 0.)
                        self.assertAlmostEqual(row["variance"], expected[mode])

    def test_readers_can_reproduce_both_printed_examples(self):
        pairs = gaussian_pairs(17)
        independent, linear = compare(.64, "independent", pairs)
        self.assertEqual([round(x, 2) for x in independent["x"]], [-1.4, -.2, .2, 1.4])
        self.assertAlmostEqual(independent["variance"], 1.)
        self.assertAlmostEqual(linear["variance"], .5392)
        fixed = compare(.64, "fixed", pairs)[0]
        self.assertEqual([round(x, 2) for x in fixed["x"]], [.04, 1.24])
        self.assertAlmostEqual(fixed["variance"], .36)
        self.assertAlmostEqual(fixed["gaussian_variance"], .3584750632671968)
        self.assertAlmostEqual(variance([10., 12.]), 1.)
        self.assertNotEqual(pairs, gaussian_pairs(29))
        self.assertEqual(pairs, gaussian_pairs(17))
        self.assertEqual(len(pairs), 4096)

    def test_invalid_or_empty_ranges(self):
        for share in (-.1,1.1,math.nan):
            with self.assertRaises(ValueError):
                coefficients(share)
        with self.assertRaises(ValueError):
            sign_cards("unknown")
