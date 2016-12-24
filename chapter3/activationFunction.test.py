import unittest
import numpy as np

import activationFunction


class TestStringMethods(unittest.TestCase):

  def testStepFunction(self):
    self.assertEqual(activationFunction.step_function(np.array([1])), 1)
    self.assertEqual(activationFunction.step_function(np.array([0])), 0)
    # just a little bit ugry...
    self.assertEqual(activationFunction.step_function(np.array([-1, 1, 0])).tolist(), np.array([0, 1, 0]).tolist())

if __name__ == '__main__':
    unittest.main()
