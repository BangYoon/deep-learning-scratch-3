# step54. Dropout & test mode

# 과대적합 일어나는 원인 
# (1) 적은 데이터 -> Data augmentation 으로 해결 
# (2) 모델 표현력이 지나치게 높음 
#    -> 가중치 감소 Weight Decay
#    -> Dropout -> Train/Test 로직 다름 
#    -> Batch normalization 


# 54.1 Direct Dropout : 일반적인 Dropout
# 학습 데이터 흘려보낼 때마다 삭제할 뉴런을 무작위로 선택
# => Ensemble Learning 과 비슷한 효과를 봄 (여러 모델을 개별 학습 후 평균내기)
import numpy as np

dropout_ratio = 0.6
x = np.ones(10)

mask = np.random.rand(10) > dropout_ratio
y = x * mask # 매회 평균 4개의 뉴런만 출력을 다음 층으로 전달함

# Train
mask = np.random.rand(*x.shape) > dropout_ratio
y = x * mask

# Test
scale = 1 - dropout_ratio # 학습 시 살아남은 뉴런의 비율 
y = x * scale # 모든 뉴런을 scaling 함으로써, 앙상블 러닝에서 여러 모델의 평균을 내는 것과 동일 효과


# 54.2 Inversed Dropout  (많이 사용됨)
# 스케일 맞추기를 Train 때 수행함 - 학습 시 1/scale 을 곱해두고 Test 때는 그대로.

# Train
scale = 1 - dropout_ratio
mask = np.random.rand(*x.shape) > dropout_ratio
y = x * mask / scale

# Test
y = x # 아무런 수행 없으니 test 수행 속도 살짝 향상됨 


if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 54.3 Test mode 추가
# dropout 사용하려면 train/test 구분 필요 -> with dezero.no_grad() 방식 사용
# core/Config수정, core/test_mode 추가, functions/dropout 추가
from dezero import test_mode
import dezero.functions as F

x = np.ones(5)
print(x)

# Train
y = F.dropout(x)
print(y)

# Test 
with test_mode():
    y = F.dropout(x)
    print(y)





