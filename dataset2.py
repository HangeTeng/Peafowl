import os
import math
import random
import numpy as np
from src.utils.h5dataset import HDF5Dataset, preprocess_linearSVM, save_subset_h5
import sys

def lsvm_gen_split(
    examples = 6000,
    features = 6000,
    nodes = 3,
    sub_examples = 5000,
    chunk = 1000,
    lsvm_gen = True,
    lsvm_split = True):
    if lsvm_gen:
        folder = "./data/SVM_{}_{}".format(examples, features)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = "{}/SVM_{}_{}.hdf5".format(folder, examples, features)
        print("generating dataset in file: {}".format(file_path))
        dataset = HDF5Dataset.new(file_path=file_path,
                                data_shape=(features, ),
                                target_shape=(),dtype=np.float32)
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
        # print(slice_features)
        for i in range(nodes):
            sub_file_path = "{}/SVM_{}_{}_{}-{}.hdf5".format(folder, 
                examples, features, i, nodes)
            indices = random.sample(range(examples), sub_examples)
            _slice = slice(slice_features * i, slice_features * (i + 1))
            save_subset_h5(dataset=dataset,file_path=sub_file_path,indices=indices,slice=_slice, slice_features = slice_features,with_targets=(i == 0), dtype=np.float32)
            print(f"generate subdataset {i} completed")
        


        dataset.close()
    temp_folder_path = folder + "/temp"
    if not os.path.exists(temp_folder_path):
        os.mkdir(temp_folder_path)
    tgt_folder_path = folder + "/tgt"
    if not os.path.exists(tgt_folder_path):
        os.mkdir(tgt_folder_path)

if __name__ == '__main__':
    import subprocess
    
    for sub_examples in [10000]:
        examples = sub_examples * 6 // 5
        for sub_features in [1]:#[500]+[2000 * i for i in range(1,5)]:
            for nodes in [5]:
                features = nodes * sub_features
                lsvm_gen_split(examples = examples,
                    features = features,
                    nodes = nodes,
                    sub_examples = sub_examples)
                subprocess.run("timeout 200 mpiexec -n {} python3 main2_.py {} {}".format(nodes + 1, examples, features), shell=True)



