# 수치 미분
# 전진차분 : f(x) = lim (f(x+h) - f(x)) / h
# 중앙차분 : f(x) = lin (f(x+h) - f(x-h)) / 2h  -> 실체 미분값과 가까움 by.Taylor series
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

# 수치 미분 구현
def numerical_diff(f, x, eps=1e-4): # h = 0.0004 (1e-4) 극한과 비슷한 값 사용
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)

# 합성 함수의 미분 : y = (e^x^2)^2 -> dy/dx 계산
def f2(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f2, x)
print(dy)

# 수치 미분의 단점 : 구현이 쉬우나, 막대한 계산량 -> 역전파 등장 (구현 복잡)







