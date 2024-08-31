# 53.1 numpy의 save 와 load
import numpy as np

# 한 개의 ndarray 저장
x = np.array([1,2,3])
np.save('step53_test.npy', x)

# 여러 개의 ndarray 저장/로드
x1 = np.array([1,2,3])
x2 = np.array([4,5,6])
np.savez('step53_test.npz', x1=x1, x2=x2)

arr = np.load('step53_test.npz')
x1 = arr['x1']
x2 = arr['x2']
print(x1, x2)
print(list(arr))

# 딕셔너리 사용
data = {'x1':x1, 'x2':x2}
np.savez('step53_test2.npz', **data)
arr = np.load('step53_test2.npz')
x1 = arr['x1']
x2 = arr['x2']
print(x1, x2)



if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero.layers import Layer, Parameter

# 53.2 Layer class의 매개변수를 평평하게 평탄화
# 계층은 Layer 안에 다른 Layer 가 들어가는 중첩 구조임. 
layer = Layer()

l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))

"""
layer 구조
|                           |
|   |p2|  |p3|  |l1 (p1)|   |
|                           |
"""
# Q. 위 같은 계층 구조의 Parameter 을 '하나의 평평한 딕셔너리'로, 즉 중첩되지 않는 딕셔너리로 뽑아내려면?
# A. Layer() 클래스에 _flatten_params() 메서드 추가. 

params_dict = {}
layer._flatten_params(params_dict)
print(params_dict) 
# {'l1/p1': variable(1), 'p2': variable(2), 'p3': variable(3)}


# 53.3 Layer 클래스의 save/load -> layers.save_weights(), load_weights()
import os
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model) # optimizer은 model의 모든 매개변수 갱신하는 역할

# 매개변수 읽기
if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0

    for x,t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update() # 매개변수 전부 갱신
        sum_loss += float(loss.data) * len(t)

    print('epoch: {}, loss: {:.4f}'.format(epoch+1, sum_loss/len(train_set)))

# 매개변수 저장하기
model.save_weights('my_mlp.npz') # loss 계속 감소 확인함 
