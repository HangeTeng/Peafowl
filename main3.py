import os, sys
from mpi4py import MPI
import numpy as np
import math
import time
from concurrent.futures import ThreadPoolExecutor
from line_profiler import LineProfiler
import random

from src.communicator.node import Node
from src.utils.h5dataset import HDF5Dataset
from src.utils.crypto.prf import PRF
from src.utils.crypto.shprg import SHPRG
from src.utils.encoder import FixedPointEncoder, mod_range

import time


class Timer:
    def __init__(self):
        self.time_points = {}
        self.currentlabel = None

    def set_time_point(self, label):
        self.time_points[label] = time.time()
        self.currentlabel = label

    def __str__(self):
        output = ""
        previous_time = None
        all_time = 0
        output += f"timepoint  |   all    | elapsed  |\n"
        for label, timestamp in self.time_points.items():
            if previous_time is not None:
                elapsed_time = timestamp - previous_time
                all_time += elapsed_time
                output += f"{label} | {all_time:.4f} s | {elapsed_time:.4f} s |\n"
            previous_time = timestamp

        return output

def main():
    # dataset
    arguments = sys.argv[1:]
    examples = int(arguments[0])
    features = int(arguments[1])
    chunk = 100
    # sub_dataset
    nodes = MPI.COMM_WORLD.Get_size() - 1
    sub_examples = examples * 5 // 6
    sub_features = features // nodes
    targets_rank = 0
    target_length = 1
    folder_path = "./data/SVM_{}_{}".format(examples, features)

    file = open("./data/log/SVM_{}_{}_log_{}.txt".format(examples, features,nodes), 'a')
    sys.stdout = file
    sys.stdout = sys.__stdout__

    secret_key = "secret_key"

    # shprg
    n = 2
    m = sub_features + target_length
    EQ = 128
    EP = 64
    q = 2**EQ
    p = 2**EP
    seedA = bytes(0x355678)

    precision_bits = 16
    encoder = FixedPointEncoder(precision_bits=precision_bits)

    shprg = SHPRG(input=n, output=m, EQ=EQ, EP=EP, seedA=seedA)

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    global_grp = global_comm.Get_group()
    client_grp = global_grp.Excl([global_size - 1])
    client_comm = global_comm.Create(client_grp)
    client_rank = None if client_comm == MPI.COMM_NULL else client_comm.Get_rank(
    )
    client_size = client_grp.Get_size()

    is_server = False
    if global_rank == global_size - 1:
        is_server = True
    server_rank = global_size - 1

    timer = Timer()

    #* initial node
    if is_server:
        node = Node(None, None, global_comm, client_comm)
        temp_dataset = []
        temp_folder_path = folder_path + "/temp"
        for i in range(client_size):
            temp_path = "{}/SVM_{}_{}_{}-{}_temp.hdf5".format(
                temp_folder_path, examples, features, i, nodes)
            temp_dataset.append(
                HDF5Dataset.new(file_path=temp_path,
                                  data_shape=(sub_features, ),
                                  target_shape=(),
                                  dtype=np.int64))
    else:
        src_path = "{}/SVM_{}_{}_{}-{}.hdf5".format(folder_path, examples,
                                                    features, global_rank,
                                                    nodes)
        src_dataset = HDF5Dataset(file_path=src_path)
        tgt_folder_path = folder_path + "/tgt"
        tgt_path = "{}/SVM_{}_{}_{}-{}_tgt.hdf5".format(
            tgt_folder_path, examples, features, global_rank, nodes)
        tgt_dataset = HDF5Dataset.new(file_path=tgt_path,
                                        data_shape=(features, ),
                                        target_shape=(),
                                        dtype=np.int64)
        node = Node(src_dataset, tgt_dataset, global_comm, client_comm)

    # print("start test...")
    timer.set_time_point("start_test")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))
    
    # random permute
    if is_server:
        permutes =[]
        for _ in range(client_size):
            all_indices = list(range(sub_examples))
            random.shuffle(all_indices)
            permutes.append(all_indices)
        # print(random_permutes)
    else:
        pass

    # sys.exit()

    # share_tras
    if is_server:
        all_deltas = [[] for _ in range(client_size)]

        def STsend_thread(rank):
        # for rank in range(client_size):
            for j in range(client_size):
                if rank == j:
                    all_deltas[rank].append(None)
                    continue
                delta = np.empty((sub_examples, n), dtype=object)
                for k in range(n):
                    _delta = node.STsend(size=sub_examples,
                                         permute=permutes[j],
                                         recver=rank,
                                         tag=j + k * 100,
                                         Sip="127.0.0.1:12233")
                    delta[:, k] = _delta
                all_deltas[rank].append(delta)

        with ThreadPoolExecutor(max_workers=client_size) as executor:
            executor.map(STsend_thread, range(client_size))
    else:
        a_s = []
        b_s = []
        for rank in range(client_size):
            if client_rank == rank:
                a_s.append(None)
                b_s.append(None)
                continue
            a = np.empty((sub_examples, n), dtype=object)
            b = np.empty((sub_examples, n), dtype=object)
            for k in range(n):
                _a, _b = node.STrecv(size=sub_examples,
                                     sender=server_rank,
                                     tag=rank + k * 100,
                                     Sip="127.0.0.1:12233")
                a[:, k] = _a
                b[:, k] = _b
            a_s.append(a)
            b_s.append(b)

    timer.set_time_point("share_tras")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))
    
    # sys.exit()    

    #* encrypted ID
    if is_server:
        id_enc = None
    else:
        prf = PRF(secret_key=secret_key)
        id_enc = np.vectorize(prf.compute)(node.src_dataset.ids[...])
    id_enc_gather = node.gather(id_enc, server_rank)

    #* server-aid PSI
    if is_server:
        final_permutes, permute_length = node.find_intersection_indices(
            id_enc_gather[:-1])
    else:
        pass
    # sys.exit()

    #* find permute
    if is_server:
        def find_permute(permute, final_permute):
            pre_permute = [0] * len(all_indices)
            for i in range(len(all_indices)):
                pre_permute[permute[i]] = final_permute[i]
            return pre_permute
        pre_permutes = []
        for i in range(client_size):
            pre_permutes.append(find_permute(permutes[i], final_permutes[i]))
        pre_permutes.append(None)

        print(pre_permutes)
        node.scatter(pre_permutes, server_rank)
    else:
        pre_permute = node.scatter(None, server_rank)
        print(pre_permute)

    # sys.exit()

    #* seeds generation
    if is_server:
        pass
    else:
        seeds = [(None if i == client_rank else np.array(
            [[k + j * 10 + i * 100 + client_rank * 1000 for k in range(n)]
             for j in range(sub_examples)]))
                 for i in range(client_size)]  #! test
        
        # seeds = [(None if i == client_rank else SHPRG.genMatrixAES128(seed=token_bytes(16),n=sub_examples,m=n,EQ=EQ) ) for i in range(client_size)]

    timer.set_time_point("server_psi")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))
    
    # sys.exit()

    # seeds share
    if is_server:
        pass
    else:
        seeds_exchanged = node.alltoall(seeds, in_clients=True)

    timer.set_time_point("seed_share")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))


    # permute and share
    if is_server:
        seeds_exchanged = None
    else:
        for i in range(client_size):
            if client_rank == i:
                continue
            seeds_exchanged[i] = (seeds_exchanged[i] - a_s[i]) % q
    seeds_share_gather = node.gather(seeds_exchanged, server_rank)

    if is_server:
        for i in range(client_size):
            for rank in range(client_size):
                if i == rank:
                    continue
                seeds_share_gather[i][rank] = (
                    seeds_share_gather[i][rank][permutes[rank]] +
                    all_deltas[i][rank]) % q
        seed1s_s = seeds_share_gather
    else:
        seed2s = b_s

    timer.set_time_point("perm_share")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))
    

    #* share
    round_examples = sub_examples // chunk + (1 if sub_examples % chunk != 0
                                              else 0)
    if is_server:
        def sharesend_thread(rank):
            for i in range(round_examples):
                recv = node.recv(source=rank, tag=i)
                temp_dataset[rank].add(data=recv[0], targets=recv[1])

        with ThreadPoolExecutor(max_workers=client_size) as executor:
            executor.map(sharesend_thread, range(client_size))
    else:
        with_targets = node.src_dataset.with_targets

        data_to_server = np.empty((chunk, sub_features), dtype=np.int64)
        targets_to_server = np.empty(
            (chunk, target_length), dtype=np.int64) if with_targets else None
        index = 0

        for i in range(round_examples):
            rest = min(sub_examples - index, chunk)
            for j in range(rest):
                perm_index = pre_permute[index + j]
                data_to_server[j] = encoder.encode(node.src_dataset.data[perm_index])
                if with_targets:
                    targets_to_server[j] = encoder.encode(node.src_dataset.targets[perm_index].reshape((1, target_length)))
            for k in range(client_size):
                if k == client_rank:
                    continue
                output_prg = mod_range(
                    shprg.genRandom(seeds[k][index:index + rest]),
                    p).astype(np.int64)
                data_to_server[:rest] -= output_prg[:, :sub_features]
                if with_targets:
                    targets_to_server[:rest] -= output_prg[:, sub_features:sub_features +
                                         target_length]
            index += rest
            if target_length == 1:
                node.send(
                    (data_to_server[:rest], targets_to_server[:rest].ravel()
                     if with_targets else None),
                    dest=server_rank,
                    tag=i)
                continue
            node.send((data_to_server[:rest],
                       targets_to_server[:rest] if with_targets else None),
                      dest=server_rank,
                      tag=i)

    timer.set_time_point("dset_share")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))
    
    # sys.exit()

    # share intersection size
    if is_server:
        pass
    else:
        permute_length = None
    permute_length = global_comm.bcast(permute_length, root=server_rank)

    # sys.exit()

    # tgt dataset server send
    round_inter = permute_length // chunk + (1 if permute_length % chunk != 0
                                             else 0)
    if is_server:
        def tgt_dataset_send(rank):
        # for rank in range(client_size):
            with_targets = (rank == targets_rank)

            data_to_client = np.empty((chunk, sub_features), dtype=np.int64)
            targets_to_client = np.empty(
                (chunk,
                 target_length), dtype=np.int64) if with_targets else None
            index = 0

            for i in range(round_inter):
                rest = min(permute_length - index, chunk)
                for j in range(rest):
                    perm_index = permutes[rank][index + j]
                    data_to_client[j] = temp_dataset[rank].data[perm_index]
                    if with_targets:
                        targets_to_client[j] = temp_dataset[rank].targets[
                            perm_index].reshape((1, target_length))
                for k in range(client_size):
                    if k == rank:
                        continue
                    output_prg = mod_range(
                        shprg.genRandom(seed1s_s[k][rank][index:index + rest]),
                        p).astype(np.int64)
                    data_to_client[:rest] += output_prg[:, :sub_features]
                    if with_targets:
                        targets_to_client[:rest] += output_prg[:, sub_features:sub_features+target_length]
                index += rest
                if target_length == 1:
                    node.send(
                        (data_to_client[:rest], targets_to_client[:rest].ravel()
                        if with_targets else None),
                        dest=rank,
                        tag=i)
                    continue
                node.send(
                    (data_to_client[:rest],
                        targets_to_client[:rest] if with_targets else None),
                    dest=rank,
                    tag=i)

        with ThreadPoolExecutor(max_workers=client_size) as executor:
            executor.map(tgt_dataset_send, range(client_size))
    else:
        index = [0] * client_size
        data = np.empty((chunk, features), dtype=np.int64)
        targets = np.empty((chunk,target_length), dtype=np.int64)
        for i in range(round_inter):
            recv = node.recv(source=server_rank, tag=i)
            rest = len(recv[0])
            # print(rest)
            for rank in range(client_size):
                if client_rank == rank:
                    data[:rest, rank * sub_features:(rank + 1) *
                         sub_features] = recv[0]
                    if rank == targets_rank:
                        targets[:rest]= recv[1].reshape((rest, target_length))
                    continue
                output_prg = mod_range(
                    shprg.genRandom(seed2s[rank][index[rank]:index[rank] + rest]),
                      p).astype(np.int64)
                data[:rest, rank * sub_features:(rank + 1) *
                     sub_features] = output_prg[:, :sub_features]
                if rank == targets_rank:
                    targets[:rest] = output_prg[:, sub_features:sub_features+target_length]
                index[rank] += rest
            if target_length == 1:
                node.tgt_dataset.add(data=data[:rest],
                                     targets=targets[:rest].ravel())
                continue
            node.tgt_dataset.add(data=data[:rest], targets=targets[:rest])

    timer.set_time_point("tgt_final ")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))
    print("intersection size:{}".format(permute_length))
    print(timer)

    file.close()


if __name__ == "__main__":
    profiler = LineProfiler()
    profiler.add_function(main)
    profiler.run('main()')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_filename = f'profile_rank_{rank}.txt'
    
    with open(output_filename, 'w') as output_file:
        profiler.print_stats(stream=output_file)

