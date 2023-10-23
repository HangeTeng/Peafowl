import os
import math
import random
import numpy as np
from src.utils.h5dataset import HDF5Dataset, preprocess_linearSVM, save_subset_h5

def lsvm_gen_split(
    examples = 60000,
    features = 6000,
    nodes = 3,
    sub_examples = 50000,
    chunk = 100,
    lsvm_gen = True,
    lsvm_split = True):
    if lsvm_gen:
        folder = "./data/SVM_{}_{}".format(examples, features)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = "{}/SVM_{}_{}.hdf5".format(folder, examples, features)
        print("generating dataset in file: {}".format(file_path))
        dataset = HDF5Dataset.empty(file_path=file_path,
                                data_shape=(features, ),
                                targets_shape=(),dtype=np.float32)
        rounds = math.ceil(examples / chunk)
        for i in range(rounds):
            print("generating dataset:%.2f%%" % (i * 100 / rounds))
            train_data, train_targets = preprocess_linearSVM(examples=chunk,
                                                                features=features)
            dataset.add(data=train_data, targets=train_targets)
        print("generate dataset completed")
        dataset.close()
    if lsvm_split:
        folder = "./data/SVM_{}_{}".format(examples, features)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = "{}/SVM_{}_{}.hdf5".format(folder, examples, features)
        dataset = HDF5Dataset(file_path=file_path)
        slice_features = features // nodes
        for i in range(nodes):
            sub_file_path = "{}/SVM_{}_{}_{}-{}.hdf5".format(folder, 
                examples, features, i, nodes)
            indices = random.sample(range(examples), sub_examples)
            _slice = slice(slice_features * i, slice_features * (i + 1))
            save_subset_h5(dataset=dataset,file_path=sub_file_path,indices=indices,slice=_slice, with_targets=(i == 0), dtype=np.float32)
        dataset.close()

if __name__ == '__main__':
    
    for examples in [12000]:
        sub_examples = examples * 5 / 6
        for features in [6000]:
            for nodes in [3]:
                lsvm_gen_split(examples = examples,
    features = 6000,
    nodes = 3,
    sub_examples = 50000)



