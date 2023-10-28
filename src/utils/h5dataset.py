import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import h5py
import os
import numpy as np
import math
import random


class HDF5Dataset(Dataset):
    allow_overwite = True

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} does not exist')
        self.file = h5py.File(file_path, 'a')
        if "data" not in self.file:
            raise ValueError("Invalid file format: missing 'data' datasets")
        self.data = self.file["data"]
        self.with_targets = False
        if "targets" in self.file:
            self.targets = self.file["targets"]
            if len(self.targets) == len(self.data):
                self.with_targets = True
        self.with_ids = False
        if "ids" in self.file:
            self.ids = self.file["ids"]
            if len(self.targets) == len(self.data):
                self.with_ids = True

    @classmethod
    def from_array(cls, file_path: str, data, targets):
        if all(
            [os.path.exists(file_path), HDF5Dataset.allow_overwite is False]):
            raise ValueError(
                "file already exists and do not allow overwriting")
        if any([data is None, targets is None]):
            raise ValueError("data or targets cannot be None")

        with h5py.File(file_path, 'w') as file:
            file.create_dataset("data",
                                data=data,
                                maxshape=(None, ) + data.shape[1:])
            file.create_dataset("targets",
                                data=targets,
                                maxshape=(None, ) + targets.shape[1:])
        return cls(file_path)

    @classmethod
    def new(cls,
              file_path: str,
              data_shape,
              target_shape,
              dtype=np.float32):
        if any([data_shape is None, target_shape is None]):
            raise ValueError("data_shape or targets_shape cannot be None")
        data = np.empty((0, ) + data_shape, dtype=dtype)
        targets = np.empty((0, ) + target_shape, dtype=dtype)
        return cls.from_array(file_path, data, targets)
    
    @classmethod
    def empty(cls, file_path: str, data_shape, targets_shape,dtype=np.float32):
        if all(
            [os.path.exists(file_path), HDF5Dataset.allow_overwite is False]):
            raise ValueError(
                "file already exists and do not allow overwriting")
        if any([data_shape is None, targets_shape is None]):
            raise ValueError("data or targets cannot be None")

        with h5py.File(file_path, 'w') as file:
            file.create_dataset("data",
                                shape=data_shape, dtype=dtype)
            file.create_dataset("targets",
                                shape=targets_shape,
                                dtype=dtype)
        return cls(file_path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index] if self.with_targets else None
        id = self.ids[index] if self.with_ids else None
        return data, target, id

    def add(self, data, targets=None, ids=None):
        self.data.resize((self.data.shape[0] + data.shape[0], ) +
                         data.shape[1:])
        self.data[-data.shape[0]:] = data
        if targets is not None:
            self.targets.resize((self.targets.shape[0] + targets.shape[0], ) +
                                targets.shape[1:])
            self.targets[-targets.shape[0]:] = targets
        if ids is not None:
            self.ids.resize((self.ids.shape[0] + ids.shape[0], ) +
                            ids.shape[1:])
            self.ids[-ids.shape[0]:] = ids

    def resize(self, data_shape, targets_shape=None, ids_shape=None):
        self.data.resize(data_shape)
        if targets_shape is not None:
            self.targets.resize(targets_shape)
        if ids_shape is not None:
            self.ids.resize(ids_shape)


    def close(self):
        self.file.close()


def get_subset(dataset, indices, slice, with_targets=False):
    '''
    Extracts a subset of data from the given dataset based on the provided indices and slice.
    Note: Slice operation flatten the data by reshaping it, keeping only the specified slice of features
    '''
    subset_data = dataset.data[indices]
    subset_data = subset_data.reshape(subset_data.shape[0], -1)[:, slice]
    subset_targets = dataset.targets[indices] if with_targets else None
    return subset_data, subset_targets


def save_subset_h5(dataset,
                   file_path,
                   indices,
                   slice,
                   slice_features,
                   with_targets=False,
                   dtype=np.float32):
    '''
    Extracts a subset of data into a HDF5 file.
    also save the indices.
    '''
    sub_dataset = HDF5Dataset.new(file_path=file_path,
                                    data_shape=(slice_features, ),
                                    target_shape=(),
                                    dtype=np.float32)
    count = 0
    for index in indices:
        subset_data, subset_targets = get_subset(
            dataset, indices=[index], slice=slice,
            with_targets=with_targets)  # rank 0 get the targets
        count += 1
        # if count % 100 == 0:
        #     print("generating sub_dataset:%.2f%%" % (count * 100 / len(indices)))
        sub_dataset.add(data=subset_data, targets=subset_targets)
    sub_dataset.file.create_dataset("ids", data=indices)
    sub_dataset.close()


def preprocess_mnist(data_dir='./data'):
    train = datasets.MNIST(data_dir, train=True, download=True)
    test = datasets.MNIST(data_dir, download=True, train=False)

    # compute normalization factors
    data_all = torch.cat([train.data, test.data]).float()
    data_mean, data_std = data_all.mean(), data_all.std()
    tensor_mean, tensor_std = data_mean.unsqueeze(0), data_std.unsqueeze(0)

    # normalize data
    train_data_norm = transforms.functional.normalize(train.data.float(),
                                                      tensor_mean, tensor_std)
    train_targets = train.targets

    return train_data_norm, train_targets


def preprocess_cifar(data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.CIFAR10(data_dir,
                             train=True,
                             download=True,
                             transform=transform)

    # normalize data
    train_data_norm = torch.stack([train[i][0] for i in range(len(train))],
                                  dim=0)
    train_targets = torch.tensor(train.targets)
    return train_data_norm, train_targets


def preprocess_linearSVM(features=6, examples=5, dtype=np.float32):
    # Set random seed for reproducibility
    np.random.seed(1)
    # Initialize x, y, w, b
    w_true = np.random.randn(features).astype(dtype)
    b_true = np.random.randn(1).astype(dtype)

    np.random.seed()
    x = np.random.randn(examples, features)
    # x = np.array([[i*100+j for j in range(features)] for i in range(examples)]) # for test
    y = np.sign(np.dot(w_true, x.T) + b_true)
    return x, y


if __name__ == '__main__':

    mnist_test = False

    cifar10_test = False

    lsvm_test = False

    lsvm_gen = True
    lsvm_split = True
    examples = 12000
    features = 18000
    chunk = 100
    nodes = 3
    sub_examples = 10000

    if mnist_test:
        train_data, train_targets = preprocess_mnist()
        file_path = "../../data/MNIST_train.hdf5"
        dataset = HDF5Dataset(file_path=file_path)
        dataset.close()

    if cifar10_test:
        train_data, train_targets = preprocess_cifar()
        file_path = "../../data/CIFAR10_train.hdf5"
        dataset = HDF5Dataset(file_path=file_path)
        dataset.close()

    if lsvm_test:
        train_data, train_targets = preprocess_cifar()
        file_path = "../../data/SVM_{}_{}.hdf5".format(examples, features)
        dataset = HDF5Dataset(file_path=file_path)
        dataset.close()

    if lsvm_gen:
        folder = "../../data/SVM_{}_{}".format(examples, features)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = "{}/SVM_{}_{}.hdf5".format(folder, examples, features)
        print("generating dataset in file: {}".format(file_path))
        dataset = HDF5Dataset.new(file_path=file_path,
                                    data_shape=(features, ),
                                    target_shape=(),
                                    dtype=np.float32)
        rounds = math.ceil(examples / chunk)
        for i in range(rounds):
            print("generating dataset:%.2f%%" % (i * 100 / rounds))
            train_data, train_targets = preprocess_linearSVM(examples=chunk,
                                                             features=features)
            dataset.add(data=train_data, targets=train_targets)
        print("generate dataset completed")
        dataset.close()

    if lsvm_split:
        folder = "../../data/SVM_{}_{}".format(examples, features)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = "{}/SVM_{}_{}.hdf5".format(folder, examples, features)
        dataset = HDF5Dataset(file_path=file_path)
        slice_features = features // nodes
        for i in range(nodes):
            sub_file_path = "{}/SVM_{}_{}_{}-{}.hdf5".format(
                folder, examples, features, i, nodes)
            indices = random.sample(range(examples), sub_examples)
            # print(indices)
            _slice = slice(slice_features * i, slice_features * (i + 1))
            save_subset_h5(dataset=dataset,
                           file_path=sub_file_path,
                           indices=indices,
                           slice=_slice,
                           slice_features = slice_features,
                           with_targets=(i == 0),
                           dtype=np.float32)
            d = HDF5Dataset(sub_file_path)  # for test
        dataset.close()
