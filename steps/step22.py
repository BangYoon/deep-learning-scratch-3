# 22. operator overload (3)
# 21.1 as_variable
# 21.2 float, int
# 21.3 __rmul__, __radd__
# 21.4 __array_priority__

import weakref
import numpy as np
import contextlib


class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):  # 18.5 mode transmition
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)

def no_grad():
  return using_config('enable_backprop', 'False')


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} type is not allowed.'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property  # shape method 를 인스턴스 변수처럼 사용할 수 있음
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
      return len(self.data)

    def __repr__(self):
      if self.data is None:
        return 'variable(None)'
      p = str(self.data).replace('\n', '\n' + ' ' * 9)
      return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1   # parent func's gen + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]   # 1. Push grads into list
            gxs = f.backward(*gys)                          # 2. Call f's backward()
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):                # 3. Save grad matching the input to x.grad
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None     # y is weakref


def as_array(y):
    if np.isscalar(y):
        return np.array(y)
    return y

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):    # if not tuple, make tuple
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])   # 1. 역전파 모드에서만 세대 필요함
            for output in outputs:
                output.set_creator(self)    # 2. 계산들의 연결도 역전파 모드에서만 필요함
            self.inputs = inputs            # inputs cnt=1
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    def backward(self, gy):
        return gy, gy

class Mul(Function):
  def forward(self, x0, x1):
    return x0 * x1
  def backward(self, gy):
    x0, x1 = self.inputs[0].data, self.inputs[1].data
    return gy * x1, gy * x0

class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        return x ** self.c
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)
def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0,x1)
def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1,x0)

def pow(x, c):
    return Pow(c)(x)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul # 첫 번째 인수가 float/int 인 경우
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow

generations = [2,0,1,4,2]
funcs = []
for g in generations:
    f = Function() # dump func
    f.generation = g
    funcs.append(f)

print([f.generation for f in funcs], end=' -> ')
funcs.sort(key=lambda x:x.generation)
print([f.generation for f in funcs])
f = funcs.pop()
print(f.generation) # 4

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad) # (2x^4)' = 8x^3

# 18.1 delete needless gradients
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print(y.grad, t.grad)   # needless
print(x0.grad, x1.grad) # only need

with no_grad():
  x = Variable(np.array(2.0))
  y = square(x)

# 21.4 좌항이 ndarray 인스턴스인 경우
x = Variable(np.array([1.0]))
y = np.array([2.0]) + x
# call ndarray instance's __add__
# want to call Variable intance's __radd__ => add __array_priority__
print(y)

x = Variable(np.array(2.0))
y = x**3
print(y)
