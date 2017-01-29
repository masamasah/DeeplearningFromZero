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

class ReluLayer(unittest.TestCase):
    def testForward(self):
        val = np.array([1, 10000, 0, -1])

        relu_layer = l.ReluLayer()

        result = relu_layer.forward(val)

        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 10000)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 0)

    def testBackword(self):
        val = np.array([1, 10000, 0, -1])

        relu_layer = l.ReluLayer()

        result = relu_layer.forward(val)

        d = np.array([100, -10, 0, 1])
        bk = relu_layer.backward(d)

        self.assertEqual(bk[0], 100)
        self.assertEqual(bk[1], -10)
        self.assertEqual(bk[2], 0)
        self.assertEqual(bk[3], 0)

class TestCompLayer(unittest.TestCase):

  def testcomp(self):
      apple = 100
      apple_num = 2
      orange = 150
      orange_num = 3
      tax = 1.1

      mul_apple_layer = l.MulLayer()
      mul_orange_layer = l.MulLayer()
      add_price_layer = l.AddLayer()
      mul_tax_layer = l.MulLayer()

      apple_price = mul_apple_layer.forward(apple, apple_num)
      orange_price = mul_orange_layer.forward(orange, orange_num)

      price = add_price_layer.forward(apple_price, orange_price)

      total_price = mul_tax_layer.forward(price, tax)

      self.assertEqual(round(total_price,0), 715)

      dprice = 1
      dprice, dtax = mul_tax_layer.backward(1)
      dapple_price, dorange_price = add_price_layer.backward(dprice)
      dapple, dapple_num = mul_apple_layer.backward(dapple_price)
      dorange, dorange_num = mul_orange_layer.backward(dorange_price)

      self.assertEqual(round(dapple, 1), 2.2)
      self.assertEqual(round(dapple_num, 0), 110)
      self.assertEqual(round(dorange, 1), 3.3)
      self.assertEqual(round(dorange_num, 0), 165)
      self.assertEqual(round(dtax, 0), 650)

if __name__ == '__main__':
    unittest.main()
