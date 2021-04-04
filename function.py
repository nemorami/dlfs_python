import numpy as np

from variable import Variable


class Function:
    def __call__(self, input: Variable):
        self.input = input
        output =  Variable(self.forward(input.data))
        output.set_creator(self)
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return 2 * self.input.data * gy


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return np.exp(self.input.data) * gy


def exp(x):
    return Exp()(x)