# 56. CNN mechanism (2)

# 56.1 3D tensor
"""
    4 2 1 2         4 0 2
  3 0 6 5 4       0 1 3 0 
1 2 3 0 3 2     2 0 1 2 2
0 1 2 3 0 5  *  0 1 2 0     =  63 55   
3 0 1 2 1       1 0 2          18 51
2 3 0 1
    입력            필터          출력 
  (C,H,W)       (C,KH,KW)    (1,OH,OW)

* 입력과 필터의 채널 수를 똑같이 맞춰야 함
* 출력 == feature map

* 특징 맵을 채널 방향으로 여러 장 갖고 싶으면? 
=> 다수의 필터(가중치) 사용하기

(C,H,W)   *  (OC,C,KH,KW)  =  (OC,OH,OW)  +  (OC,1,1) = (OC,OH,OW) 
              = 필터 OC개                        편향 
"""

# 56.3 미니배치 처리
# 신경망 학습은 여러 입력 데이터를 하나의 단위(미니배치)로 묶어 처리한다.
# 미니배치 처리하려면, 각 층 흐르는 데이터를 '4차원 텐서' 로 취급하기
""" N개 데이터로 이루어진 미니배치의 합성곱 예
(N, C,H,W) * (OC,C,H,W) = (N, OC,OH,OW) + (OC,1,1) = (N, OC,OH,OW)
 N개 입력                      N개 출력       

* N = batch_size               
"""

# 56.4 풀링층 
""" 
풀링은 가로,세로 공간을 작게 만드는 연산.  ex) max pooling, avg pooling 
Pooling 특징
    * 학습하는 매개변수가 없다. (학습할 게 없다..)
    * 채널 수가 변하지 않는다. (채널마다 독립적으로 계산되므로, 입/출력 채널 그대로.)
    * 미세한 위치 변화에 영향을 덜 받는다. (입력 데이터의 미세한 차이에 강건하다.)
 """