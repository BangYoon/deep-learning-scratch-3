# 51. MNIST training
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt


# 51.1 MNIST dataset
import dezero
train_set = dezero.datasets.MNIST(train=True, transform=None)
test_set = dezero.datasets.MNIST(train=False, transform=None)

print(len(train_set))
print(len(test_set))

x, t= train_set[0]
print(type(x), x.shape) 
# <class 'numpy.ndarray'> (1, 28, 28) : 1채널(grayscale), 28x28 pixel image
print(t) 
# 5 : 해당 이미지 정답 5

plt.imshow(x.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show()
print('label:', t)


# 이제 input data 전처리 해야함. 
# 사실 이 f(x) 기능이 MNIST dataset 에는 기본 탑재. 
def f(x):
    x = x.flatten() # (1,28,28) -> (784,)
    x = x.astype(np.float32)
    x /= 255.0      # 모든 값의 범위: 0.0 ~ 1.0
    return x

train_set = dezero.datasets.MNIST(train=True, transform=f)
test_set = dezero.datasets.MNIST(train=False, transform=f)


# 51.2 MNIST 학습하기
import dezero
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F
from dezero import DataLoader

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))

# 51.3 모델 개선하기 
# ReLU 사용하기 - accuracy 98% 얻음 
model = MLP((hidden_size,hidden_size, 10), activation=F.relu)

optimizer = optimizers.SGD().setup(model) 
# optimizer은 model의 모든 매개변수 갱신하는 역할

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0,0 

    for x,t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y,t)
        acc = F.accuracy(y,t)

        model.cleargrads()
        loss.backward()
        optimizer.update() # 매개변수 전부 갱신

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('train_loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0,0 
    with dezero.no_grad():
        for x,t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y,t)
            acc = F.accuracy(y,t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))

