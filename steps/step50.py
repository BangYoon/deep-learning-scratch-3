# 미니배치 뽑아주는 DataLoader
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math

import dezero
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F
# 50.1 Dezero.Dataloader 추가

# 50.2 Dataloader 사용
from dezero.datasets import Spiral
from dezero import DataLoader

batch_size = 10
max_epoch = 1

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)
        break

    for x, t in test_loader:
        print(x.shape, t.shape)
        break


# 50.3 accuracy 함수 구현 -> Dezero/functions
# accuracy(y, t) : y=predicted, t=label
y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
t = np.array([1,2,0])
acc = F.accuracy(y,t)
print(acc)


# 50.4 Spiral Dataset 학습 코드
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

# data_size = len(train_set)
# max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # index = np.random.permutation(data_size)
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader: # 훈련용 미니배치
        y = model(x)
        loss = F.softmax_cross_entropy(y,t)
        acc = F.accuracy(y,t) # 훈련 데이터의 인식 정확도
        
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)


    # print per epoch
    print('epoch %d, train loss %.4f, acc: %.4f' % (epoch + 1, sum_loss/len(train_set), sum_acc/len(train_set)))


    sum_loss, sum_acc = 0,0
    with dezero.no_grad(): # 기울기 불필요 모드 = test 모드
        for x,t in test_loader: 
            y = model(x)
            loss = F.softmax_cross_entropy(y,t)
            acc = F.accuracy(y,t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('\ttest loss: %.4f, acc: %.4f' % (sum_loss/len(train_set), sum_acc/len(train_set)))
