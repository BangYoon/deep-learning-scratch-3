import os
import gzip
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dezero.utils import get_file, cache_dir
from dezero.transforms import Compose, Flatten, ToFloat, Normalize

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
    

# =============================================================================
# MNIST-like dataset: MNIST / CIFAR /
# =============================================================================
class MNIST(Dataset):

    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(),
                                     Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


# =============================================================================
# Sequential datasets: SinCurve, Shapekspare
# =============================================================================
class SinCurve(Dataset):

    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


class Shakespear(Dataset):

    def prepare(self):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        file_name = 'shakespear.txt'
        path = get_file(url, file_name)
        with open(path, 'r') as f:
            data = f.read()
        chars = list(data)

        char_to_id = {}
        id_to_char = {}
        for word in data:
            if word not in char_to_id:
                new_id = len(char_to_id)
                char_to_id[word] = new_id
                id_to_char[new_id] = word

        indices = np.array([char_to_id[c] for c in chars])
        self.data = indices[:-1]
        self.label = indices[1:]
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char

