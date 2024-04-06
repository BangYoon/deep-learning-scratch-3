import unittest
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} type is not allowed.'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()             # pop last func
            x, y = f.input, f.output    # bring I/O
            x.grad = f.backward(y.grad) # call backward method

            if x.creator is not None:
                funcs.append(x.creator) # input의 creator 함수를 리스트에 추가

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
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
        x = self.input.data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(y):
    if np.isscalar(y):
        return np.array(y)
    return y

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

# python -m unittest steps/step10.py
# or unittest.main() -> python steps/step10.py

