from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L

class Model(Layer):
    # Layer 클래스 기능 상속받으며, 시각화 메서드 추가됨 
    def plot(self, *inputs, to_file='./model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


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

# model = TwoLayerNet(hidden_size, 1)


# 범용적인 완전연결계층 신경망 !! - Multi-Layer Perceptron 
class MLP(Model):
    # fc_output_sizes = (10,10,1) -> linear layer 3층
    # activation 지정 가능 
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes): # full-connect
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer) # self.l1 = ... 형태로 layer 추가할 수 없으니까! 
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


