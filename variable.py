import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = np.ones_like(data)
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        while f:
            f.input.grad = f.backward(f.output.grad)
            f = f.input.creator

