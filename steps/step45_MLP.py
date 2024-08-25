# 45. 다른 Layer 를 담는 Layer
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Model, MLP
import dezero.functions as F
import dezero.layers as L

# dataset 생성 
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# hyper-parameter 설정
lr = 0.2
max_iter = 10000
hidden_size = 10

# model 정의
class TwoLayerNet(Model):
    # Layer class를 편리하게 사용하여 
    # 여러 linear layer 갖는 모델 정의하기
    # Layer(또는 Layer을 상속받는 Model)을 상속하여 모델 전체를 하나의 'class'로 정의하는 방법

    def __init__(self, hidden_size, out_size):
        # __init__ : 필요한 Layer 생성하여 self.l1 = ... 형태로 설정
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        # forward : 추론을 수행 
        y = F.sigmoid_simple(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(hidden_size, 1)

# 학습
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)

model = MLP((10, 20, 30, 1)) #4층

