import sys
from mpi4py import MPI
import numpy as np
from node import Node



if __name__ == "__main__":
    # dataset
    examples = 6
    features = 60
    chunk = 100

    # sub_dataset
    nodes = 3
    sub_examples = 5

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    size = global_comm.Get_size()

    global_grp = global_comm.Get_group()

    print(global_rank)
    client_grp = global_grp.Excl([size-1])
    print(global_rank)
    client_comm = global_comm.Create(global_grp)
    client_rank = client_comm.Get_rank()

    print(global_rank)

    server_rank = size-1
    is_server = False
    if global_rank == server_rank:
        is_server = True

    if is_server:
        print("yes")
        node = Node(None,None,global_comm,client_comm)
    else:
        src_path = "../data/SVM_{}_{}_{}-{}.hdf5".format(
        examples, features, global_rank, nodes)
        src_dataset = HDF5Dataset(file_path=src_path)
        tgt_path = "../data/SVM_{}_{}_{}-{}_share.hdf5".format(
                examples, features, global_rank, nodes)
        tgt_dataset = HDF5Dataset.empty(file_path=tgt_path,data_shape=(features,),targets_shape=(),dtype=np.int64)
    
        node = Node(None,None,global_comm,client_comm)

    print("start testing...")

