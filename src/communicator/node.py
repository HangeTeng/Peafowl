import sys
if __name__ == "__main__":
    sys.path.append('../../')

from mpi4py import MPI
import numpy as np
import time
import random

from src.utils.h5dataset import HDF5Dataset
from src.communicator.STComm import Sender, Receiver



def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"{func.__name__} took {elapsed_time:.2f} ms to execute")
        return result

    return wrapper


class Node():
    def __init__(self, src_dataset: HDF5Dataset, tgt_dataset: HDF5Dataset,
                 global_comm: MPI.Comm, client_comm: MPI.Comm):
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.global_comm = global_comm
        self.client_comm = client_comm
        self.totalDataSent = 0
        self.totalDataRecv = 0
        self.STSender = Sender(4)
        self.STRecver = Receiver(4)
    


    # @ timer
    def send(self, data, dest, tag=0, in_clients=False):
        comm = self.client_comm if in_clients else self.global_comm
        comm.send(data, dest=dest, tag=tag)
        self.totalDataSent += self.get_size_recursive(data) / 1024 / 1024

    # @ timer
    def recv(self, source, tag=0, in_clients=False):
        comm = self.client_comm if in_clients else self.global_comm
        data = comm.recv(source=source, tag=tag)
        self.totalDataRecv += self.get_size_recursive(data) / 1024 / 1024
        return data

    def gather(self, send_data, root_rank, in_clients=False):
        comm = self.client_comm if in_clients else self.global_comm
        recv_data = comm.gather(send_data, root_rank)
        if root_rank == comm.Get_rank():
            self.totalDataRecv += self.get_size_recursive(recv_data) / 1024 / 1024
            self.totalDataSent += self.get_size_recursive(send_data) / 1024 / 1024
        else:
            self.totalDataSent += self.get_size_recursive(send_data) / 1024 / 1024
        return recv_data

    def scatter(self, send_data, root_rank, in_clients=False):
        comm = self.client_comm if in_clients else self.global_comm
        recv_data = comm.scatter(send_data, root_rank)
        if root_rank == comm.Get_rank():
            self.totalDataSent += self.get_size_recursive(send_data) / 1024 / 1024
            self.totalDataRecv += self.get_size_recursive(recv_data) / 1024 / 1024
        else:
            self.totalDataRecv += self.get_size_recursive(recv_data) / 1024 / 1024
        return recv_data

    def alltoall(self, send_data, in_clients=False):
        comm = self.client_comm if in_clients else self.global_comm
        recv_data = comm.alltoall(send_data)
        self.totalDataSent += self.get_size_recursive(send_data) / 1024 / 1024
        self.totalDataRecv += self.get_size_recursive(recv_data) / 1024 / 1024
        return recv_data
    
    # @timer
    def STsend(self,
               size,
               permute,
               recver=None,
               in_clients=False,
               tag=0,
               p=1 << 128,
               Sip="127.0.0.1:12222",
               ot_type=1,
               num_threads=2):
        comm = self.client_comm if in_clients else self.global_comm
        sessionHint = str(tag) if recver is None else str(
            comm.Get_rank()) + "_" + str(recver) + "_" + str(tag)
        result = self.STSender.run(size=size,
                        sessionHint=sessionHint,
                        permute=permute,
                        p=p,
                        Sip=Sip,
                        ot_type=ot_type,
                        num_threads=num_threads)
        self.totalDataSent += self.STSender.getTotalDataSent() / 1024 / 1024
        self.totalDataRecv += self.STSender.getTotalDataRecv() / 1024 / 1024
        return result

    # @timer
    def STrecv(self,
               size,
               sender=None,
               in_clients=False,
               tag=0,
               p=1 << 128,
               Sip="127.0.0.1:12222",
               ot_type=1,
               num_threads=2):
        comm = self.client_comm if in_clients else self.global_comm
        sessionHint = str(tag) if sender is None else str(sender) + "_" + str(
            comm.Get_rank()) + "_" + str(tag)
        result = self.STRecver.run(size=size,
                            sessionHint=sessionHint,
                            p=p,
                            Sip=Sip,
                            ot_type=ot_type,
                            num_threads=num_threads)
        self.totalDataSent += self.STSender.getTotalDataSent() / 1024 / 1024
        self.totalDataRecv += self.STSender.getTotalDataRecv() / 1024 / 1024
        return result[0],result[1]


    def get_size_recursive(self, obj):
        size = sys.getsizeof(obj)
        
        if isinstance(obj, (np.ndarray)):
            return size + obj.nbytes
        elif isinstance(obj, (int, float, bool, str)):
            return size
        elif isinstance(obj, (list, tuple)):
            return size + sum(self.get_size_recursive(item) for item in obj)
        elif isinstance(obj, dict):
            return size + sum(self.get_size_recursive(key) + self.get_size_recursive(value) for key, value in obj.items())
        
        return size


    def __split_array(self, arr, split_num=2):
        shape = arr.shape
        arr_split = np.empty(shape + (split_num, ), dtype=np.uint64)
        for i in range(split_num):
            arr_split[..., i] = (arr >> (i * 64)) & 0xFFFFFFFFFFFFFFFF
        return arr_split

    def __combine_array(self, arr, arr_split):
        split_num = arr_split.shape[-1]
        arr.fill(0)
        for i in range(split_num):
            arr += (arr_split[..., i].astype(object) << (i * 64))
        return arr

    def __Send(self, data, dest, tag=0, in_clients=False, split_num=2):
        comm = self.client_comm if in_clients else self.global_comm
        _data = self.__split_array(data, split_num=split_num)
        comm.Send(_data, dest=dest, tag=tag)
        self.totalDataSent += self.get_size_recursive(_data) / 1024 / 1024

    def __Recv(self, data, source, tag=0, in_clients=False, split_num=2):
        comm = self.client_comm if in_clients else self.global_comm
        _data = np.empty(data.shape + (split_num, ), dtype=np.uint64)
        comm.Recv(_data, source=source, tag=tag)
        self.totalDataRecv += self.get_size_recursive(_data) / 1024 / 1024
        self.__combine_array(data, _data)

    def find_intersection_indices(self, arrays_list):
        if len(arrays_list) < 2:
            return []

        intersection_set = set(arrays_list[0])
        for arr in arrays_list[1:]:
            intersection_set &= set(arr)

        intersection_list = list(intersection_set)
        random.seed(1) #! test
        random.shuffle(intersection_list)
        # print(intersection_list)

        length_intersection = len(intersection_list)

        intersection_indices = []
        for arr in arrays_list:
            indices = []
            if isinstance(arr, list):
                for intersection_element in intersection_list:
                    indices.append(arr.index(intersection_element))
            elif isinstance(arr, np.ndarray):
                for intersection_element in intersection_list:
                    indices.extend(list(np.where(arr == intersection_element)[0]))
            all_indices = list(range(len(arr)))
            non_common_indices = list(set(all_indices) - set(indices))    
            random.shuffle(non_common_indices)
            indices = indices + non_common_indices
            intersection_indices.append(indices)

        return intersection_indices, length_intersection


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
    comm_size = global_comm.Get_size()

    global_grp = global_comm.Get_group()
    client_grp = global_grp.Excl([comm_size - 1])
    client_comm = global_comm.Create(global_grp)
    client_rank = client_comm.Get_rank()

    is_server = False
    if global_rank == comm_size - 1:
        is_server = True
    server_rank = comm_size - 1

    if is_server:
        node = Node(None, None, global_comm, client_comm)
    else:
        src_path = "../../data/SVM_{}_{}_{}-{}.hdf5".format(
            examples, features, global_rank, nodes)
        src_dataset = HDF5Dataset(file_path=src_path)
        tgt_path = "../../data/SVM_{}_{}_{}-{}_share.hdf5".format(
            examples, features, global_rank, nodes)
        tgt_dataset = HDF5Dataset.empty(file_path=tgt_path,
                                        data_shape=(features, ),
                                        targets_shape=(),
                                        dtype=np.int64)



        node = Node(src_dataset, tgt_dataset, global_comm, client_comm)

    print("start testing...")

    size = 4
    # gather
    data_to_gather = np.array([10]*size) + global_rank
    gathered_data = node.gather(data_to_gather, server_rank)
    if is_server:
        print(f"Gathered data: {gathered_data}")

    # scatter
    data_to_scatter = None
    if is_server:
        data_to_scatter = np.array([100]*size) + global_rank
    received_data = node.scatter(data_to_scatter, server_rank)
    print(f"Scattered data: {received_data}")

    # alltoall
    data_to_exchange = np.array([1000]*size) + global_rank
    exchanged_data = node.alltoall(data_to_exchange, in_clients=True)
    print(f"Exchanged data: {exchanged_data}")

    data_size = 6
    permute = range(data_size - 1, -1, -1)

    Sip = "127.0.0.1:12222"
    STData = None
    # for j in range(comm_size - 1):
    for j in range(1):
        for i in range(comm_size - 1):
            if is_server:
                STData = node.STsend(size=data_size,
                                    permute=permute,
                                    recver=i,
                                    tag=j,
                                    Sip=Sip)
            elif global_rank == i:
                STData,STData2 = node.STrecv(size=data_size,
                                    sender=server_rank,
                                    tag=j,
                                    Sip=Sip)
            # print(str(i)+"_"+str(j)+" complete!")
    if is_server:
        print("rank " +str(global_rank) +" sent: "+ str(node.STSender.getTotalDataSent()))
        print("rank " +str(global_rank) +" recv: "+ str(node.STSender.getTotalDataRecv()))
    else:
        print("rank " +str(global_rank) +" sent: "+ str(node.STRecver.getTotalDataSent()))
        print("rank " +str(global_rank) +" recv: "+ str(node.STRecver.getTotalDataRecv()))
    


