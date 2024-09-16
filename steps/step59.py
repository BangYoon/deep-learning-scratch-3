# 59. RNN을 활용한 시계열 데이터 처리
"""
그동안은 feed forward 구조의 신경망 - 데이터를 순방향으로만 입력해준다. 입력 신호만으로 출력을 결정한다. 
RNN(Recurrent Neural Network) 
    - 순환 신경망. RNN의 출력은 자기 자신에게 피드백된다. 
    - '상태'를 가짐.
    - 데이터가 입력되면 '상태' 갱신, 출력에 영향도 줌.
"""

# 59.1 RNN 계층 구현
""" 입력 x_t, 은닉 상태 h_t를 출력하는 RNN
h_t = tanh(h_t-1 * W_h + x_t * W_x + b)
        - W_x : x_t를 은닉 상태 h로 변환하기 위한 가중치 
        - W_h : RNN 출력을 다음 시각의 출력으로 변환하기 위한 가중치    
        - h_t : 시각 t에서의 출력 h_t. 다음 계층의 입력이면서, 다음 시각의 자기 자신의 입력.
"""

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.dataloaders
import dezero.datasets
import dezero.layers as L
import dezero.optimizers

rnn = L.RNN(10) # 은닉층의 크기만 지정
x = np.random.rand(1, 1)
h = rnn(x)
print(h.shape, h)


# 59.2 RNN 모델 구현
# - Linear 계층을 써서 RNN 계층의 은닉 상태를 출력으로 변환하기!

from dezero import Model
import dezero.functions as F

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size) # RNN의 은닉 상태 입력받아서 모델의 최종 출력을 계산함. 

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y
    

seq_data = [np.random.randn(1,1) for _ in range(1000)] # dummy 시계열 데이터
xs = seq_data[0:-1]
ts = seq_data[1:] # 정답 데이터: xs 보다 한 단계 앞선 데이터

model = SimpleRNN(10, 1)

loss, cnt = 0, 0
for x,t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)

    cnt +=1

    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break


# 59.3 '연결'을 끊어주는 메서드
# - 이전 학습에서 사용한 계산 그래프까지 기울기가 흐르지 못하도록 은닉 상태 변수에서의 계산 '연결'을 끊어야함.
# -> Variable 클래스에 unchain(), unchain_backward() 추가 


# 59.4 사인파 예측
import numpy as np
import dezero
import matplotlib.pyplot as plt

train_set = dezero.datasets.SinCurve(train=True)
print(len(train_set))
print(train_set[0])
print(train_set[1])
print(train_set[2])

# draw graph
xs = [example[0] for example in train_set]
ts = [example[1] for example in train_set]
plt.plot(np.arange(len(xs)), xs, label='xs')
plt.plot(np.arange(len(ts)), ts, label='ts')
plt.show()


# 사인파 학습하기
max_epoch = 100
hidden_size = 100
bptt_length = 30
train_set = dezero.datasets.SinCurve(train=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

# 학습 시작
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0,0

    for x,t in train_set:
        x = x.reshape(1,1)
        y = model(x)
        loss += F.mean_squared_error(y,t)
        count+=1

        # Truncated BPTT 타이밍 조절 (계산 연결 끊기)
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch+1, avg_loss))


# Test
xs = np.cos(np.linspace(0, 4*np.pi, 1000))
model.reset_state() 
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1,1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()