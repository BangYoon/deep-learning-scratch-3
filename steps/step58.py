if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.models import VGG16

# 58. 대표 CNN - VGG16 구현
# con2d, pooling 사용!

# 58.1 VGG16 구현

# 용어 설명
# - 3x3 Conv 64 = 커널 크기가 3x3이고, 출력 채널 수가 64개
# - pool/2 = 2x2 풀링
# - Linear 4096 = 출력 크기가 4096 인 완전연결계층

# VGG16 특징
# - 3x3 conv 사용
# - conv 채널 수는 풀링하면 2배로 중가 (64->128->256->512)
# - 완전연결계층에서는 dropout 사용
# - 활성화함수로 ReLU 사용


model = VGG16(pretrained=True)

x = np.random.rand(1,3,224,224).astype(np.float32)
model.plot(x, to_file='vgg.pdf')


# 58.3 trained VGG16 사용하기
import dezero
from PIL import Image

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
image_path = './zebra.jpg'
img = Image.open(image_path)
img.show()

x = VGG16.preprocess(img)
# print(type(x), x.shape)   # (3, 224, 224)
x = x[np.newaxis] # 배치용 축 추가 -> (1, 3, 224, 224)

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file='vgg_zebra.pdf')
labels=dezero.datasets.ImageNet.labels()
print(labels[predict_id])