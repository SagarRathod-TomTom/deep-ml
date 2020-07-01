import unittest
import tests

if __name__ == "__main__":

    test_suite = unittest.TestLoader().discover(tests.__name__)

    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(test_suite)