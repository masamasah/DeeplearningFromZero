import unittest
import numpy as np

import Layers as l

class TestMulLayer(unittest.TestCase):

  def testForward(self):
      apple = 100
      apple_num = 2
      tax = 1.1

      mul_apple_layer = l.MulLayer()
      mul_tax_layer = l.MulLayer()

      apple_price = mul_apple_layer.forward(apple, apple_num)
      price = mul_tax_layer.forward(apple_price, tax)

      self.assertEqual(int(price), 220)

  def testBackword(self):
      apple = 100
      apple_num = 2
      tax = 1.1
      mul_apple_layer = l.MulLayer()
      mul_tax_layer = l.MulLayer()

      apple_price = mul_apple_layer.forward(apple, apple_num)
      price = mul_tax_layer.forward(apple_price, tax)

      dprice = 1
      dapple_price, dtax_price = mul_tax_layer.backward(dprice)
      dapple, dapple_num = mul_apple_layer.backward(dapple_price)

      self.assertEqual(round(dapple,2), 2.2)
      self.assertEqual(round(dapple_num,0), 110)
      self.assertEqual(round(dtax_price,0), 200)

class TestAddLayer(unittest.TestCase):

  def testForward(self):
      apple = 100
      orange = 200

      mul_layer = l.AddLayer()
      price = mul_layer.forward(apple, orange)

      self.assertEqual(int(price), 300)

  def testBackword(self):
      apple = 100
      orange = 200

      mul_layer = l.AddLayer()
      price = mul_layer.forward(apple, orange)

      dapple, dorange = mul_layer.backward(price)

      self.assertEqual(dapple, 300)
      self.assertEqual(dorange, 300)


if __name__ == '__main__':
    unittest.main()
