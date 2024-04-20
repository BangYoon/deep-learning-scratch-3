# 28. 함수 최적화

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
# # import dezero's simple_core explicitly
# import dezero
# if not dezero.is_simple_core:
#     from dezero.core_simple import Variable
#     from dezero.core_simple import setup_variable
#     setup_variable()


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)
# 기울기는 함수 출력 y값을 가장 크게 늘려주는 방향
# 기울기 * -1 은 y값을 가장 작게 줄여주는 방향

# 경사하강법
# 국소적으로 함수의 출력을 가장 크게 하는 방향(기울기)을 구하고 일정 거리 이동해서 다시 기울기 구하기
# -> 최댓값 혹은 회솟값에 접근 가능함
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 1000

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

