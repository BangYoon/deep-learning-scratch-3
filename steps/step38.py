# 37. 텐서 다루기
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)


# 38.2 Variable에서 reshape 사용하기
x = np.random.rand(1,2,3)
print(x)
y = x.reshape((2,3))    # tuple
y = x.reshape([2,3])    # list
y = x.reshape(2,3)      # 인수 그대로 받기
print(y)


# 38.3 행렬의 전치
x = np.array([[1,2,3], [4,5,6]])
y = np.transpose(x)
print(y)

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.transpose(x)
y = x.T
y.backward()
print(x.grad)