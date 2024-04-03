# 수동 역전파
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # gradient = differential val

class Function:
     def __call__(self, input):
         x = input.data
         y = self.forward(x)
         output = Variable(y)
         self.input = input # save the input
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

# 순전파
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

# 역전파
# ((e^x^2)^2)' = (e^(2x^2))' = 4x e^x^2
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad) #3.29744 = step04 의 수치미분 값과 유사함 





