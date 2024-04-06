# Auto back-propagation
# if run forward-propagation, automatically run back-propagation

# Define-by-Run : connect DL calculations when start running
# Square > Exp > Square
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # gradient = differential val
        self.creator = None

    def set_creator(self, func):
        self.creator = func

class Function:
     def __call__(self, input):
         x = input.data
         y = self.forward(x)
         output = Variable(y)

         #output's creator == function
         output.set_creator(self)
         #save I/O
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

# 순전파 forward-propagation
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# check Backwards.. in reverse order
assert y.creator == C
assert y.creator.input == b # call C > __call__ > save I/O
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# 역전파 back-propagation
y.grad = np.array(1.0)
# implement (b > C > y)
C = y.creator
b = C.input
b.grad = C.backward(y.grad)

# implement (a > B > b)
B = b.creator
a = B.input
a.grad = B.backward(b.grad)

# implement (x > A > a)
A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad) #similar to numerical_diff

# Variable 인스턴스의 creator == None 이면 역전파 중단됨







