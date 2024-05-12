# 39. 합계 함수
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([1,2,3,4,5,6]))
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)


# 39.3 axis 지정, keepdims
x = np.array([[1,2,3],[4,5,6]])
y = np.sum(x, axis=0)
print(y)
y = np.sum(x, keepdims=True)
print(y)
print(y.shape)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
y = x.sum(keepdims=True)
print(y.shape)