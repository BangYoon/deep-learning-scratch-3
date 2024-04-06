# 재귀에서 반복문으로

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # gradient = differential val
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    """
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward() # self.creator == None 인 변수를 찾을 때까지 반복됨 : recursive
    """
    def backward(self):
        # 처리해야 할 함수들을 funcs 리스트에 차례로 집어넣음
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
        output = Variable(y)

        # output's creator == function
        output.set_creator(self)
        # save I/O
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
        gx = 2 * x * gy # x^2 의 미분은 2x / gy: 출력쪽에서 전해지는 미분값
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy # e^x 의 미분은 e^x
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# back-propagation
y.grad = np.array(1.0)
y.backward()
print(x.grad)





