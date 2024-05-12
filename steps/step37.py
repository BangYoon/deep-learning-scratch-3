# 37. 텐서 다루기
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

# x : scalar
x = Variable(np.array(1.0))
y = F.sin(x)
print(y)

# x : tensor
x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.sin(x)
print(y)

# tensor 사용 시의 역전파
x = Variable(np.array([[1,2,3], [4,5,6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
y = F.sum(t)

y.backward(retain_grad=True)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)

