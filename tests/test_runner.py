import unittest
from gaim.backend import call_seed
from gaim.run import validate_config


class ProtocolTest(unittest.TestCase):
    def test_seed_is_stable_and_condition_specific(self):
        self.assertEqual(call_seed(5, 'q1', 'adaptive', 1), call_seed(5, 'q1', 'adaptive', 1))
        self.assertNotEqual(call_seed(5, 'q1', 'adaptive', 1), call_seed(5, 'q1', 'independent', 1))

    def test_test_split_cannot_be_used_as_pilot(self):
        with self.assertRaisesRegex(ValueError, 'reserved'):
            validate_config({'run_kind': 'pilot', 'split': 'test'})


if __name__ == '__main__':
    unittest.main()
