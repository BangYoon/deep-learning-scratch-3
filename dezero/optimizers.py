# Optimizer로 수행하는 매개변수 갱신
# : 매개변수 개인 작업을 모듈화하고 쉽게 다른 모듈로 대체할 수 있는 구조
import numpy as np
import math

class Optimizer:
    def __init__(self):
        self.target = None # Model 또는 Layer class
        self.hooks = [] # 전처리 함수 (옵션)

    def setup(self, target): # 매개변수를 갖는 클래스(Model 또는 Layer)를 인스턴스 변수인 target으로 설정 
        self.target = target
        return self

    def update(self): # 모든 매개변수 갱신 (grad=None 인 매개변수는 건너뜀)
        # None 외의 매개변수 리스트에 모아둠
        params = [p for p in self.target.params() if p.grad is not None]

        # 전처리 (옵션)
        for f in self.hooks:
            f(params)

        # 매개변수 갱신
        for param in params:
            self.update_one(param)

    def update_one(self, param): # 구체적인 매개변수 갱신 내용 수행 
        # Optimizer의 자식 클래스에서 재정의해야 함
        raise NotImplementedError()

    def add_hook(self, f): # 원하는 전처리가 있으면 전처리 함수 추가함 
        self.hooks.append(f)


# 경사하강법 으로 매개변수 갱신
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__() # target model과 전처리 함수 설정
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key] # v=속도
        v *= self.momentum # momentum=가속도 관한 값 (보통 0.9 로 해서 서서히 속도 감속시킴)
        v -= self.lr * param.grad.data
        param.data += v


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = np #cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)