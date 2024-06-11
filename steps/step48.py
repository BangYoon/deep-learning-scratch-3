# 48. 다중 클래스 분류
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# 1. hyper params 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# 2. data 읽기 / 모델, optimizer 생성
x, t = dezero.datasets.get_spiral(train=True) # 입력 데이터 x, 정답 데이터 t
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

print(x.shape)
print(t.shape)
print(x[10], t[10])
print(x[110], t[110])

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # 3. data index 섞기
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 4. mini batch 생성
        batch_index = index[i* batch_size:(i+1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 5. 기울기 산출 / 매개변수 갱신
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    # 6. epoch 별 학습 결과 출력
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch+1, avg_loss))


model.plot(x)