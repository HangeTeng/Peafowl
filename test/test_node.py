import sys
sys.path.append('../')

from mpi4py import MPI
import numpy as np

from src.communicator.node import Node
from src.utils.h5dataset import HDF5Dataset


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
    client_grp = global_grp.Excl([size-1])
    client_comm = global_comm.Create(global_grp)
    client_rank = client_comm.Get_rank()

    is_server = False
    if global_rank == size-1 :
        is_server = True
    server_rank = size-1

    if is_server:
        node = Node(None,None,global_comm,client_comm)
    else:
        src_path = "../data/SVM_{}_{}_{}-{}.hdf5".format(
                examples, features, global_rank, nodes)
        src_dataset = HDF5Dataset(file_path=src_path)
        tgt_path = "../data/SVM_{}_{}_{}-{}_share.hdf5".format(
                examples, features, global_rank, nodes)
        tgt_dataset = HDF5Dataset.empty(file_path=tgt_path,data_shape=(features,),targets_shape=(),dtype=np.int64)
    
        node = Node(src_dataset,tgt_dataset,global_comm,client_comm)

    print("start testing...")

    # 测试 gather 数据
    data_to_gather = np.array([10]*size) + global_rank
    gathered_data = node.gather(data_to_gather, server_rank)
    if is_server:
        print(f"Gathered data: {gathered_data}")

    # 测试 scatter 数据
    data_to_scatter = None
    if is_server:
        data_to_scatter = np.array([100]*size) + global_rank
    received_data = node.scatter(data_to_scatter, server_rank)
    print(f"Scattered data: {received_data}")

    # 测试 alltoall 数据
    data_to_exchange = np.array([1000]*size) + global_rank
    exchanged_data = node.alltoall(data_to_exchange)
    print(f"Exchanged data: {exchanged_data}")