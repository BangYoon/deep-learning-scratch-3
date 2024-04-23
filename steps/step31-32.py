# 31. 고차 미분 (이론)
"""
- 계산 그래프의 '연결'은 순전파 계산 시 만들어짐
31.1 역전파 계산 시에도 '연결' 만들어지면 고차 미분 자동 계산 가능해짐!

Ex)
class Sin(Function):
    ...
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)     # 이 코드를 계산 그래프화.
        return gx

x -> sin -> y
y.backward()    : y의 x에 대한 미분 구해짐

x -> cos -> * gy -> gx
gx = gy * np.cos(x) : sin 의 역전파 코드임
gx.backward()   : gx의 x에 대한 미분 구해짐, gx는 원래 y=sin(x)의 미분이므로
                : y의 x에 대한 2차 미분임
"""

