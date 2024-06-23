import os
import gzip
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from dezero.utils import get_file, cache_dir
# from dezero.transforms import Compose, Flatten, ToFloat, Normalize

class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train # flag - train/test

        # pre-processing
        self.transform = transform  # 입력 데이터 하나에 대한 전처리
        self.target_transform = target_transform  # 레이블 하나에 대한 전처리
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()  # 자식 클래스에서는 prepare()가 데이터 준비 작업 하도록 구현.

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            # return self.data[index], None   # 정답 없을 때 None 반환
            return self.transform(self.data[index]), None
        else:
            # return self.data[index], self.label[index]
            return self.transform(self.data[index]), \
                   self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


# =============================================================================
# Toy datasets
# =============================================================================
def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int64)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


class BigData(Dataset): # BigData 초기화 시 데이터 아직 읽지 않고, 데이터 접근 시에만 읽기.
    def __getitem__(index):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('label/{}.npy'.format(index))
        return x, t

    def __len__():
        return 1000000