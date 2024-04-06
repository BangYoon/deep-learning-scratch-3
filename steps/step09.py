# Simplify funcs - method 1 ~ 4
import numpy as np

class Variable:
    def __init__(self, data): # method 4 : allow only ndarray for data
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} type is not allowed.'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # method 3 : y.grad = np.grad(1) 생략하기
        if self.grad is None:
            self.grad = np.ones_like(self.data) # make all elements to 1, keeping up data type

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
        # output = Variable(y)  # forward 에서 x^2 결과는 float32 임 -> as_array(y) 필요함
        output = Variable(as_array(y))
        output.set_creator(self)    # output's creator == function
        self.input = input          # save I/O
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

"""
def square(x):
    f = Square()
    return f(x)
def exp(x):
    f = Exp()
    return f(x)
"""
def square(x): # simplifying method 1
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(y):
    if np.isscalar(y):
        return np.array(y)
    return y

x = Variable(np.array(0.5))
# a = square(x)
# b = exp(a)
# y = square(b)
y = square(exp(square(x))) #simplifying method 2

# back-propagation
# y.grad = np.array(1.0) # skip by method 3
y.backward()
print(x.grad)

x = Variable(np.array(1.0)) # OK
x = Variable(None)  # OK
# x = Variable(1.0)   # error





