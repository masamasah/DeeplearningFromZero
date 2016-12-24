import unittest
import numpy as np

import activationFunction


class TestStringMethods(unittest.TestCase):

  def testStepFunction(self):
    self.assertEqual(activationFunction.step_function(np.array([1])), 1)
    self.assertEqual(activationFunction.step_function(np.array([0])), 0)
    # just a little bit ugry...
    self.assertEqual(activationFunction.step_function(np.array([-1, 1, 0])).tolist(), np.array([0, 1, 0]).tolist())

  def testSigmoidFunction(self):
    self.assertEqual(activationFunction.sigmoid(0), 0.5)
    self.assertEqual(activationFunction.sigmoid(np.inf), 1)
    self.assertEqual(activationFunction.sigmoid(-np.inf), 0)
    # just a little bit ugry...
    self.assertEqual(activationFunction.sigmoid(np.array([-np.inf, 0, np.inf])).tolist(), np.array([0, 0.5, 1]).tolist())

  def testReluFunction(self):
    self.assertEqual(activationFunction.relu(-1.0), 0)
    self.assertEqual(activationFunction.relu(0), 0)
    self.assertEqual(activationFunction.relu(1.0), 1.0)
    self.assertEqual(activationFunction.relu(5.0), 5.0)
    # just a little bit ugry...
    self.assertEqual(activationFunction.relu(np.array([-1.0, 0, 1.0, 5.0])).tolist(), np.array([0, 0, 1.0, 5.0]).tolist())

if __name__ == '__main__':
    unittest.main()
