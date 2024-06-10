# 47. softmax, cross entropy
# 다중 클래스 분류
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# 47.1 슬라이스 조작 함수 - get_item()
# 47.2 softmax
model = MLP((10, 3))

x = np.array([[0.2, -0.4]])
y = model(x)
print(y)

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = softmax1d(y) # total sum 1
print(y)
print(p)

# 47.3 cross entropy error

