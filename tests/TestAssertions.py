import unittest

from relaxations.interval import Interval
from tests.RelaxationTestCase import RelaxationTestCase


class TestAssertions(RelaxationTestCase):

    def test_interval_soundness(self):
        self.assertSound(Interval(-1.0, 1.0), [Interval(-1.0, 1.0)], lambda x: x)

    def test_approximate_interval_soundness(self):
        self.assertSound(Interval(-1.0, 1.0), [Interval(-1.0 - 1e-9, 1 + 1e-9)], lambda x: x)

    def test_upper_bound_violation(self):
        with self.assertRaises(AssertionError):
            self.assertSound(Interval(-1.0, 1.0), [Interval(-1.0, 1.0 + 1e-8)], lambda x: x)

    def test_lower_bound_violation(self):
        with self.assertRaises(AssertionError):
            self.assertSound(Interval(-1.0, 1.0), [Interval(-1.0 - 1e-8, 1.0)], lambda x: x)


if __name__ == '__main__':
    unittest.main()
