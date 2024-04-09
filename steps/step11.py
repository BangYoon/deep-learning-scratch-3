"""How to compute gradient
1. numerical differentiation    : f（x+h）+f（x-h）/2h
2. symbolic differentiation     : using math formula
3. automatic differentiation    : using chain rule (1)forward mode (2)reverse mode
"""
# 11.1 Revision of Function class
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
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

# TODO : Change the I/O of __call__ method - allows multiple I/O
class Function:
    """
    def __call__(self, input):
        x = input.data                 # 1. Pop data out of the box
        y = self.forward(x)            # 2. Compute in forward()
        output = Variable(as_array(y)) # 3. Push data into Variable
        output.set_creator(self)       # 4. Mark oneself as a creator
        self.input = input
        self.output = output
        return output
    """
    def __call__(self, inputs):
        xs = [x.data for x in inputs]  # list comprehension
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

"""
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

def square(x): # simplifying method 1
    return Square()(x)

def exp(x):
    return Exp()(x)
    
x = Variable(np.array(0.5))
y = square(exp(square(x))) 
y.backward()
print(x.grad)
"""

def as_array(y):
    if np.isscalar(y):
        return np.array(y)
    return y

class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)

def add(xs):
    return Add()(xs)


xs = [Variable(np.array(2.0)), Variable(np.array(3.0))]
ys = add(xs)
print(ys[0].data)
