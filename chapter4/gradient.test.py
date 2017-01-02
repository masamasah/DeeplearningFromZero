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

  def testNumericalGradient(self):
      def func(x):
          return x[0]**2 + x[1]**2

      ans = gradient.numerical_gradient(func, np.array([3.0, 4.0]))
      roundedVal = [round(x, 4) for x in ans.tolist()]
      self.assertEqual(roundedVal, [6.0000, 8.0000])

      ans = gradient.numerical_gradient(func, np.array([3.0, 0.0]))
      roundedVal = [round(x, 4) for x in ans.tolist()]
      self.assertEqual(roundedVal, [6.0000, 0.0000])

      ans = gradient.numerical_gradient(func, np.array([0.0, 2.0]))
      roundedVal = [round(x, 4) for x in ans.tolist()]
      self.assertEqual(roundedVal, [0.0000, 4.0000])

  def testGradient(self):
      def func(x):
          return x[0]**2 + x[1]**2

      init_x = np.array([-3.0, 4.0])
      ans = gradient.gradient_descent(func, init_x=init_x, lr=0.1, step_num=100)
      self.assertEqual(ans.tolist(), [-6.111107928998789e-10, 8.148143905314271e-10])

if __name__ == '__main__':
    unittest.main()
