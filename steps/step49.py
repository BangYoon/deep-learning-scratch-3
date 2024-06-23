# 49. Dataset class와 pre-processing
# 거대 데이터를 하나의 ndarray 인스턴스로 처리하면 모든 원소를 한번에 메모리에 올려야하는 문제.
# -> 데이터셋 전용 클래스 Dataset class인 만들어서 전치리할 수 있는 구조 추가할 예정.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 49.1 Dataset class 구현 -> dezero/dataset.py
import dezero
import math
import numpy as np
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F

train_set = dezero.datasets.Spiral(train=True)
print(train_set[0], train_set[1])
print(len(train_set))

batch_index = [0,1,2] # mini-batch 용, 0~2번째 데이터 꺼냄.
batch = [train_set[i] for i in batch_index]
# batch = [(data_0, label_0), (data_1, lable_1), (data_2, label_2)]

# 이 배치를 신경망에 입력하려면 하나의 ndarray 인스턴스로 변환해야함.
x = np.array([example[0] for example in batch])
t = np.array([example[1] for example in batch])
print(x, t)


# 49.4 학습 코드
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0
    for i in range(max_iter):
        # mini batch 꺼내기
        batch_index = index[i * batch_size:(i+1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        # 이 배치를 신경망에 입력하려면 하나의 ndarray 인스턴스로 변환해야함
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    # print per epoch
    avg_loss = sum_loss/ data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))


# 49.5 data pre-processing -> dataset.py
# 입력 데이터 전처리 예시
def f(x):
    y = x / 2.0
    return y
train_set = dezero.datasets.Spiral(transform=f)

# 입력 데이터 전처리 예시 2
from dezero import transforms
f = transforms.Normalize(mean=0.0, std=2.0)
train_set = dezero.datasets.Spiral(transforms=f)

# 여러 전처리 연달아 수행 가능!
f = transforms.Compose([transforms.Normalize(mean=0.0, std=2.0),
                        transforms.AsType(np.float64)])
train_set = dezero.datasets.Spiral(transforms=f)




