# 52. use GPU 
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt


# 52.1 - cupy 설치 및 사용법

# 52.2 - cuda 모듈
# dezero.cuda 에 as_numpy, as_cupy 추가
# dezero.core의 class Variable 에 to_cpu, to_gpu 추가
# dezero.layers의 class Layer 에 to_cpu, to_gpu 추가
# dataloader parameter로 gpu 추가, to_cpu, to_gpu 추가



import mlx.core as mx



# 52.3 GPU 로 MNIST 학습하기
import time
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

max_epoch = 5
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)

model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model) # optimizer은 model의 모든 매개변수 갱신하는 역할


# GPU mode
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()
    print('gpu')

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x,t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update() # 매개변수 전부 갱신

        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start 
    print('epoch: {}, loss: {:.4f}, time: {:.4f}[sec]'.format(epoch+1, sum_loss/len(train_set), elapsed_time))

