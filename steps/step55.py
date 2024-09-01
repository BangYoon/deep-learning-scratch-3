# 55. CNN mechanism (1) 

# 55.1 CNN 신경망 구조
# Linear -> ReLU 
# Conv -> ReLU -> (Pool) 계층 구조로 대채됨
# 출력에 가까워지면 이전과 같이 Linear -> ReLU 조합이 사용됨

# 55.2 Conv : 합성곱 연산 == 필터 연산 == 커널 연산 
""" 합성곱 연산의 예 (Conv2d)
    1 2 3 0 
    0 1 2 3     2 0 1
    3 0 1 2  *  0 1 2   =  15 16   +   3    = 18 19
    2 3 0 1     1 0 2       6 15               9 18
    입력데이터      필터      출력데이터     편향 
"""

# 55.3 Padding
# 합성곱층의 주요 처리 전에 입력데이터 주위에 고정값(가령 0) 을 채우기
# 목적: 출력 크기 조정하기 위해서. 합성곱 연산 거칠 때마다 공간 축소되면 더 이상 합성곱 연산 못할 수 있으므로. 
""" (4,4) 입력데이터에 폭 1짜리 패딩 적용 예 
   0 0 0 0 0 0 
   0 1 2 3 0 0
   0 0 1 2 3 0     2 0 1      7 12 10  2
   0 3 0 1 2 0  *  0 1 2   =  4 15 16 10  
   0 2 3 0 1 0     1 0 2      10 6 15  6             
   0 0 0 0 0 0                8 10  4  3
    (6,6) 입력      필터         출력데이터   => 출력이 (4,4) 입력 형상을 유지하게 됨!!
"""

# 55.4 Stride (보폭) : 필어 움직이는 간격 
# stride 크게 하면 출력 데이터 작아짐

# 55.5 출력 크기 계산 방법
def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

H,W = 4,4       # 입력
kh, kw = 3,3    # 커널 형상
sh, sw = 1,1    # stride
ph, pw = 1,1    # padding 

OH = get_conv_outsize(H, kh, sh, ph)
OW = get_conv_outsize(W, kw, sw, pw)
print(OH, OW) # 4,4 

