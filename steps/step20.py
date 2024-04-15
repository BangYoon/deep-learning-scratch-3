# 20. operator overload (1) 
# mul() -> * / add() -> +   

import weakref
from IPython.display import YouTubeVideo
import numpy as np

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} type is not allowed.'.format(type(data)))
        self.data = data
        # 19.1 변수 이름 지정 
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0     # Add generation

    def __len__(self):
      return len(self.data)

    # 객체를 문자열로 representation 
    def __repr__(self):
      if self.data is None:
        return 'variable(None)'
      p = str(self.data).replace('\n', '\n' + ' '*9)
      return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1   # parent func's gen + 1

    def __mul__(self, other):
      return mul(self, other) 

    @property # shape method 를 인스턴스 변수처럼 사용할 수 있음 
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

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 16.3 Add add_func() in Variable's backward()
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

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):    # if not tuple, make tuple
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])   # 1. 역전파 모드에서만 세대 필요함
            for output in outputs:
                output.set_creator(self)    # 2. 계산들의 연결도 역전파 모드에서만 필요함 
            self.inputs = inputs    # inputs cnt=1
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Config:
    enable_backprop = True  # 역전파 활성 모드


def as_array(y):
    if np.isscalar(y):  # == if not isinstance(y, np.ndarray):
        return np.array(y)
    return y

# 20.1 Mul
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
        x = self.inputs[0].data     # change to inputs
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data     # change to inputs
        return np.exp(x) * gy

def mul(x0, x1):
  return Mul()(x0, x1)

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


# 18.5 mode transmition using "with"

import contextlib
"""
 with open('sample.txt', 'w') as f:
  f.write('hello world!!')

@contextlib.contextmanager  # decorator - 문맥 판단하는 함수 
def config_test():
  print('start')  # 전처리
  try:
    yield     # with 블록 들어갈 때 전처리 실행, 블록 나올 때 후처리 실행됨 
  finally:
    print('done') # 후처리

with config_test():
  print('process...')
"""

@contextlib.contextmanager  
def using_config(name, value):
  old_value = getattr(Config, name)  # 전처리
  setattr(Config, name, value)
  try:
    yield     # with 블록 들어갈 때 전처리 실행, 블록 나올 때 후처리 실행됨 
  finally:
    setattr(Config, name, old_value)

"""
with using_config('enable_backprop','False'):
  # 이 안에서만 역전파 비활성 모드 - 순전파만 실행함  
  x = Variable(np.array(2.0))
  y = square(x)
"""

def no_grad(): # with using_config 매번 쓰기 귀찮으니까!! 
  return using_config('enable_backprop', 'False')

with no_grad():
  x = Variable(np.array(2.0))
  y = square(x)


# 19.
x = Variable(np.array(([[1,2,3],[4,5,6]])))
print(x.shape) #shape() 대신 shape 
print(len(x))
print(x)


# 20.
a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = add(mul(a,b), c)
y.backward()
print(y)
print(a.grad, b.grad) # 2.0 3.0
print(c.grad) # 1.0
print(y.grad) # None

y = a * b
print(y) 


# 20. operator overload 다른 방식 
Variable.__mul__ = mul
Variable.__add__ = add 

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))
y = a * b + c
y.backward()
print(y)
print(a.grad, b.grad)









