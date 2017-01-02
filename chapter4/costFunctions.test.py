import unittest
import numpy as np

import costFunctions as cf

class TestStringMethods(unittest.TestCase):


  def testMeanSquredError(self):
      t = [0,0,2,0,0,0,0,0,0,0]
      y = [0,0,0,0,0,0,0,0,0,0]

      self.assertEqual(cf.mean_squared_error(np.array(y), np.array(t)), 2)

      t = [0,0,1,0,0,0,0,0,0,0]
      y = [0,0,0,0,0,0,0,0,0,0]

      self.assertEqual(cf.mean_squared_error(np.array(y), np.array(t)), 0.5)

  def testCrossEntropyError(self):
      t = [0,0,1,0,0,0,0,0,0,0]
      y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

      val = cf.cross_entropy_error(np.array(y), np.array(t))
      roundedVal = round(val, 4)
      self.assertEqual(roundedVal, 0.5108)


if __name__ == '__main__':
    unittest.main()
