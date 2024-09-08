# 57. conv2d, pooling func
# conv 를 그대로 구현하면 for 중첩코드, 느림 - im2col 사용해서 간단 구현할 것.

# 57.1 im2col 에 의한 전개
# im2col 
# - 커널 적용할 영억을 꺼내서 한 줄로 Reshape -> 2D tensor 로 변환
# - 입력데이터를 한 줄로 '전개'하는 함수 (커널 계산 편하도록 펼쳐주기)

# im2col(x, kernel_size, stride=1, pad=0, to_matrix=True)

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F

# 57.2 con2d 구현
x1 = np.random.rand(1,3,7,7) # batch_size = 1
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)

print(col1.shape) #(9,75)
# (7,7) * (5,5) = (3,3) == (5,5)필터짜리 9개  
# 9개의 각 데이터는 5x5xC = 5x5x3 = 75

x2 = np.random.rand(10,3,7,7) # batch_size = 10
kernel_size = (5,5)
stride = (1,1)
pad = (0,0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)

print(col2.shape) #(90,75) 
# (5,5) 필터짜리 9개 * 10


from dezero.utils import pair
print(pair(1))
print(pair((1,2)))


from dezero import Variable

N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3,3)

x = Variable(np.random.rand(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()

# (N, C,H,W) * (OC,C,H,W) = (N, OC,OH,OW) + (OC,1,1) = (N, OC,OH,OW)
print(y.shape) # (1,5,15,15) * (8,5,3,3) = (1,8,15,15)
print(x.grad.shape) # (1,5,15,15)


# 57.3 layers 에 Con2d 구현

# 57.4 pooling 함수 구현 : fucntions_conv.py 에 pooling_simple
# con2d_simple과 마찬가지로 im2col 써서 데이터 전개. 
# but, 풀링은 채널 방향과는 독립적임
"""
1 2 3 0  0 0 6 5  7 2 1 2      1 2 0 1      2    2 4  4 6   7 4
0 1 2 4  4 2 4 3  0 1 0 4   -> 0 0 4 2 ->   4 -> 3 4  3 3   4 6
1 0 4 2  3 0 1 0  3 0 6 2      7 2 0 1      7   
3 2 0 1  2 3 3 1  4 2 4 5      3 0 2 4      4
                               6 5 4 3      6
                               1 2 0 4      4
                               1 0 3 2      3
                               3 0 2 3      3
                               3 0 4 2      4
                               4 2 0 1      4
                               1 0 3 1      3
                               6 2 4 5      6                            
"""
