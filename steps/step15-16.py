# 15-16. Complex computation graph (Topology : graph connected shape)
"""
x > A > a > B > b > D > y
          > C > c >

=> Back-propagation order = D > B > C > A

Problem.
funcs = [D] -> pop D
        [B, C] -> pop B
        [B, A] -> pop A : error.. need to give priority

=> use Topological Sort algorithm
    - node connection
    - record generation of each func
    ex) b,c > D > y
        - b gen = 2
        - c gen = 3
        - D gen = 3 = (input's max gen) = max(b, c)
        - y gen = 4 = (parent func's gen) + 1
"""
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} type is not allowed.'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0     # Add generation

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1   # parent func's gen + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs] # 1. Push grads into list
            gxs = f.backward(*gys)                      # 2. Call f's backward()
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):            # 3. Save grad matching the input to x.grad
                # x.grad = gx     # TODO : remove overwrite x.grad
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        self.generation = max([x.generation for x in inputs])

        ys = self.forward(*xs)
        if not isinstance(ys, tuple):    # if not tuple, make tuple
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


def as_array(y):
    if np.isscalar(y):  # == if not isinstance(y, np.ndarray):
        return np.array(y)
    return y

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data     # change to inputs
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data     # change to inputs
        return np.exp(x) * gy

def square(x): # simplifying method 1
    return Square()(x)

def exp(x):
    return Exp()(x)

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy       # partial derivative -> TOOD : fix Variable backward()

def add(*xs):
    return Add()(*xs)


x0 = Variable(np.array(2.0))
x1 = Variable(np.array(3.0))
ys = add(x0, x1)
ys.backward()
print(ys.data, x0.grad, x1.grad)

z = add(square(x0), square(x1))
z.backward()
print(z.data)
print(x0.grad)  # 2x0
print(x1.grad)  # 2x1

x = Variable(np.array(3.0))
y = add(x,x)
y.backward()
print(x.data, y.data, x.grad)

x.cleargrad()
y = add(add(x,x), x)
y.backward()
print(x.grad)

generations = [2,0,1,4,2]
funcs = []
for g in generations:
    f = Function() # dump func
    f.generation = g
    funcs.append(f)

print([f.generation for f in funcs])
funcs.sort(key=lambda x:x.generation)
print([f.generation for f in funcs])

f = funcs.pop()
print(f.generation)