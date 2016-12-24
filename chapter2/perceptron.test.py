import unittest

import perceptron

class TestStringMethods(unittest.TestCase):

  def testAnd(self):
    self.assertEqual(perceptron.AND(1,1), 1)
    self.assertEqual(perceptron.AND(0,1), 0)
    self.assertEqual(perceptron.AND(1,0), 0)
    self.assertEqual(perceptron.AND(0,0), 0)

  def testOr(self):
    self.assertEqual(perceptron.OR(1,1), 1)
    self.assertEqual(perceptron.OR(0,1), 1)
    self.assertEqual(perceptron.OR(1,0), 1)
    self.assertEqual(perceptron.OR(0,0), 0)

  def testNAND(self):
    self.assertEqual(perceptron.NAND(1,1), 0)
    self.assertEqual(perceptron.NAND(0,1), 1)
    self.assertEqual(perceptron.NAND(1,0), 1)
    self.assertEqual(perceptron.NAND(0,0), 1)

  def testXOR(self):
    self.assertEqual(perceptron.XOR(1,1), 0)
    self.assertEqual(perceptron.XOR(0,1), 1)
    self.assertEqual(perceptron.XOR(1,0), 1)
    self.assertEqual(perceptron.XOR(0,0), 0)

if __name__ == '__main__':
    unittest.main()
