import unittest
import numpy as np

import gradient

class TestStringMethods(unittest.TestCase):

  def testMeanSquredError(self):
      def func(x):
          return 0.01 * x ** 2 + 0.1 * x

      self.assertEqual(round(gradient.numerical_diff(func, 10), 4), 0.300)
      self.assertEqual(round(gradient.numerical_diff(func, 0), 4), 0.100)
      self.assertEqual(round(gradient.numerical_diff(func, -10), 4), -0.100)




if __name__ == '__main__':
    unittest.main()
