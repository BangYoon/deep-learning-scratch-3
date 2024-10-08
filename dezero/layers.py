import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter
from dezero import cuda
from dezero.utils import pair
import os 

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
    
    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        # self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key:param.data for key, param in params_dict.items()
                      if param is not None}
        
        try:
            np.savez_compressed(path, **array_dict)
        except(Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

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


class Con2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, 
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels      # 입력 채널 수 
        self.out_channels = out_channels    # 출력 채널 수 
        self.kernel_size = kernel_size  
        self.stride = stride
        self.pad = pad
        self.dtype = dtype      # 초기화할 가중치의 데이터 타입 

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C*KH*KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d_simple(x, self.W, self.b, self.stride, self.pad)
        return y
    

class RNN(Layer):
    def __init__(self, hidden_size, in_size=None): # in_size=None 이면 은닉 크기만 정해놓고, 입력 크기는 들어오는 데이터에 따라 자동 구현하겠다는 뜻.
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)              # x2h : 입력 x에서 은닉 상태 h로 변환하는 완전연결계층.
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True) # h2h : 이전 은닉 상태에서 다음 은닉 상태로 변환하는 완전연결계층. RNN 편향 하나이므로 애는 nobias
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))

        self.h = h_new
        return h_new
    

