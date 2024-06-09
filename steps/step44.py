# 43. 신경망
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L

# dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1) # 데이터 생성에 sin 함수 이용

# 가중치 초기화 -> Linear 인스턴스가 맡음!!
# I, H, O = 1, 10, 1
# W1 = Variable(0.01 * np.random.randn(I, H))
# b1 = Variable(np.zeros(H))
# W2 = Variable(0.01 * np.random.randn(H, O))
# b2 = Variable(np.zeros(O))

l1 = L.Linear(10) # 출력 크기 설정
l2 = L.Linear(1)

def predict(x):
    y = l1(x)           # y = F.linear_simple(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = l2(y)           # y = F.linear_simple(y, W2, b2)
    return y

lr = 0.2
iters = 10000

# 신경망 학습
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads() # W1.cleargrad() & b1.cleargrad()
    l2.cleargrads() # W2.cleargrad() & b2.cleargrad()

    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
            # W1.data -= lr * W1.grad.data & b1.data -= lr * b1.grad.data
            # W2.data -= lr * W2.grad.data & b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)
