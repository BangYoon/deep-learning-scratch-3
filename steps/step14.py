# 14. Use of variables repeatedly
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
    if np.isscalar(y):
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
print(ys.data)
print(x0.grad)
print(x1.grad)

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
