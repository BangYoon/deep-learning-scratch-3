# 46.3 SGD 클래스를 사용한 문제 해결
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# hyper-parameter 설정
lr = 0.2
max_iter = 10000
hidden_size = 10

# model 정의
model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)
# optimizer = optimizers.SGD(lr).setup(model) 와 동일하다

# 학습
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()     # -> model -> MLP -> L -> W.grad, b.grad 구함

    optimizer.update()  # -> params (W, b).grad.data update
    if i % 1000 == 0:
        print(loss)
