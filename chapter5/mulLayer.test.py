import unittest
import numpy as np

import mulLayer as ml

class TestMulLayer(unittest.TestCase):

  def testForward(self):
      apple = 100
      apple_num = 2
      tax = 1.1

      mul_apple_layer = ml.MulLayer()
      mul_tax_layer = ml.MulLayer()

      apple_price = mul_apple_layer.forward(apple, apple_num)
      price = mul_tax_layer.forward(apple_price, tax)

      self.assertEqual(int(price), 220)




if __name__ == '__main__':
    unittest.main()
