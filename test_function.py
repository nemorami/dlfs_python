from unittest import TestCase
from function import *


class TestFunction(TestCase):
    def test_forward(self):
        x = Variable(0.5)
        y = square(exp(square(x)))

        #y.grad = np.array(1.0)
        y.backward()
        self.assertEqual(round(x.grad, 5), 3.29744)



