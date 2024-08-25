import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter
from dezero import cuda

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value): #인스턴스 변수 설정 시 호출되는 특수 메서드
        # 이름이 name인 인스턴스 변수에 값으로 value를 전달함 
        if isinstance(value, (Parameter, Layer)): # Parameter 은 Variable 상속
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name] # _params 에서 name(string) 꺼내고 그 name에 해당하는 객체를 obj로 꺼냄

            if isinstance(obj, Layer):  # obj가 Layer 인스턴스면
                yield from obj.params() # obj.params 호출 -> Layer 속 Layer 재귀적으로 꺼낼 수 있음
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()


# layer.p1 = Parameter(np.array(1))
# for name in layer._params:
#   print(name, layer.__dict__[name])  =>  p1 variable(1)


# 선형 변환 클래스
"""
class Linear_origin(Layer):  #입력 크기, 출력 크기
    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()

        I, O = in_size, out_size
        W_data = np.random.randn(I,O).astype(dtype) * np.sqrt(1/I) # 가중치 초기값은 무작위로 설정해야 함
        self.W = Parameter(W_data, name='W') # 가중치
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(0, dtype=dtype), name='b') # 편향

    def forward(self, x): # 선형 변환 구현
        y = F.linear(x, self.W, self.b)
        return y
"""

class Linear(Layer):  #가중치를 나중에 (forward)에 생성함으로써 입력 크기 in_size를 자동으로 결정하기
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        # W_data = np.random.randn(I,O).astype(dtype) * np.sqrt(1/I)
        self.W = Parameter(None, name='W') # 가중치
        if self.in_size is not None: # in_size 없으면 나중으로 연기
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b') # 편향

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x): # 선형 변환 구현
        # data를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear_simple(x, self.W, self.b)
        return y

