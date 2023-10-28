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

    # output
    file = open(
        "./data/log/SVM_{}_{}_log_{}.txt".format(examples, features, nodes),
        'a')
    sys.stdout = file
    sys.stdout = sys.__stdout__

    secret_key = "secret_key"

    # shprg
    n = 8
    m = sub_features + target_length
    EQ = 128
    EP = 64
    q = 2**EQ
    p = 2**EP
    seedA = bytes(0x355678)
    shprg = SHPRG(input=n, output=m, EQ=EQ, EP=EP, seedA=seedA)

    # encoder
    precision_bits = 16
    encoder = FixedPointEncoder(precision_bits=precision_bits)

    # Communicator
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

    # thread
    max_workers = 100
    server_max_worker = max_workers * 3

    # other 
    timer = Timer()

    #* initial node
    if is_server:
        node = Node(None, None, global_comm, client_comm)
        temp_dataset = []
        temp_prg_dataset = []
        temp_folder_path = folder_path + "/temp"
        for rank in range(client_size):
            temp_path = "{}/SVM_{}_{}_{}-{}_temp.hdf5".format(
                temp_folder_path, examples, features, rank, nodes)
            temp_dataset.append(
                HDF5Dataset.empty(
                    file_path=temp_path,
                    data_shape=(sub_examples, sub_features),
                    targets_shape=(sub_examples, ) if target_length == 1 else
                    (sub_examples, target_length),
                    dtype=np.int64))
            temp_prg_path = "{}/SVM_{}_{}_{}-{}_temp_prg.hdf5".format(
                temp_folder_path, examples, features, rank, nodes)
            temp_prg_dataset.append(
                HDF5Dataset.new(
                    file_path=temp_prg_path,
                    data_shape=(sub_features, ),
                    target_shape=() if target_length == 1 else (target_length),
                    dtype=np.int64))
    else:
        src_path = "{}/SVM_{}_{}_{}-{}.hdf5".format(folder_path, examples,
                                                    features, global_rank,
                                                    nodes)
        src_dataset = HDF5Dataset(file_path=src_path)
        tgt_folder_path = folder_path + "/tgt"
        tgt_path = "{}/SVM_{}_{}_{}-{}_tgt.hdf5".format(
            tgt_folder_path, examples, features, global_rank, nodes)
        tgt_dataset = HDF5Dataset.new(
            file_path=tgt_path,
            data_shape=(features, ),
            target_shape=() if target_length == 1 else (target_length),
            dtype=np.int64)
        node = Node(src_dataset, tgt_dataset, global_comm, client_comm)

    # print("start test...")
    timer.set_time_point("start_test")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))


    # random permute
    if is_server:
        permutes = []
        for _ in range(client_size):
            random.seed(1)
            all_indices = list(range(sub_examples))
            random.shuffle(all_indices)
            permutes.append(all_indices)
        # print(permutes)
    else:
        pass


    # share_tras
    if is_server:
        all_deltas = np.empty((client_size, client_size, sub_examples, n),
                              dtype=object)

        def STsend_thread(args):
            rank, dset_rank, input_dim = args
            if rank == dset_rank:
                return
            all_deltas[rank, dset_rank, :, input_dim] = node.STsend(
                size=sub_examples,
                permute=permutes[dset_rank],
                recver=rank,
                tag=dset_rank + input_dim * 100,
                Sip="127.0.0.1:12233")

        with ThreadPoolExecutor(max_workers=server_max_worker) as executor:
            task_args = [(rank, dset_rank, input_dim)
                         for dset_rank in range(client_size)
                         for input_dim in range(n)
                         for rank in range(client_size)]
            # print(task_args)
            executor.map(STsend_thread, task_args)
        # print(all_deltas[0][1][0][0])
    else:
        a_s = np.empty((client_size, sub_examples, n), dtype=object)
        b_s = np.empty((client_size, sub_examples, n), dtype=object)

        def STrecv_thread(args):
            dset_rank, input_dim = args
            if client_rank == dset_rank:
                return
            a_s[dset_rank, :,
                input_dim], b_s[dset_rank, :, input_dim] = node.STrecv(
                    size=sub_examples,
                    sender=server_rank,
                    tag=dset_rank + input_dim * 100,
                    Sip="127.0.0.1:12233")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            task_args = [(dset_rank, input_dim)
                         for dset_rank in range(client_size)
                         for input_dim in range(n)]
            executor.map(STrecv_thread, task_args)
        # if client_rank == 0:
        #     permute = [2, 3, 4, 0, 1]
        #     print((a_s[1][2][0]-b_s[1][0][0])%q)

    timer.set_time_point("share_tras")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))
    
    

    #* encrypted ID
    if is_server:
        id_enc = None
    else:
        prf = PRF(secret_key=secret_key)
        id_enc = np.vectorize(prf.compute)(node.src_dataset.ids[...])
    id_enc_gather = node.gather(id_enc, server_rank)

    #* server-aid PSI
    if is_server:
        final_permutes, inter_length = node.find_intersection_indices(
            id_enc_gather[:-1])
    else:
        inter_length = None
    inter_length = global_comm.bcast(inter_length, root=server_rank)

    #* find permute
    if is_server:

        def find_permute(permute, final_permute):
            pre_permute = [0] * len(all_indices)
            for i in range(len(all_indices)):
                pre_permute[permute[i]] = final_permute[i]
            return pre_permute

        pre_permutes = []
        for rank in range(client_size):
            pre_permutes.append(find_permute(permutes[rank], final_permutes[rank]))
        pre_permutes.append(None)

        node.scatter(pre_permutes, server_rank)
    else:
        pre_permute = node.scatter(None, server_rank)

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
        for rank in range(client_size):
            if client_rank == rank:
                continue
            seeds_exchanged[rank] = (seeds_exchanged[rank] - a_s[rank]) % q
    seeds_share_gather = node.gather(seeds_exchanged, server_rank)

    if is_server:
        for recv_rank in range(client_size):
            for send_rank in range(client_size):
                if recv_rank == send_rank:
                    continue
                seeds_share_gather[recv_rank][send_rank] = (
                    seeds_share_gather[recv_rank][send_rank][permutes[send_rank]] +
                    all_deltas[recv_rank][send_rank]) % q
        seed1s_s = seeds_share_gather
        # print(seed1s_s[0][1][0][0])
    else:
        seed2s = b_s
        # print(b_s[1][0][0])

    timer.set_time_point("perm_share")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))

    #* share
    round_examples = sub_examples // chunk + (1 if sub_examples % chunk != 0
                                              else 0)
    round_inter = inter_length // chunk + (1 if inter_length % chunk != 0
                                             else 0)
    if is_server:
        for rank in range(client_size):
            temp_prg_dataset[rank].resize(data_shape = (inter_length,sub_features), targets_shape = (inter_length,) if target_length == 1 else
                    (inter_length, target_length))

        def sharerecv_thread(args):
            rank,round = args
            with_targets = (rank == targets_rank)
            index = round * chunk
            rest = min(sub_examples - index, chunk)
            
            recv = node.recv(source=rank, tag=round)

            temp_dataset[rank].data[index:index + rest] = recv[0]
            if with_targets:
                temp_dataset[rank].targets[index:index + rest]=recv[1].ravel() if target_length == 1 else recv[1]

        def tgt_prg_cal_thread(args):
            rank,round = args
            with_targets = (rank == targets_rank)

            data_to_client = np.zeros((chunk, sub_features), dtype=np.int64)
            targets_to_client = np.zeros(
                (chunk,
                 target_length), dtype=np.int64) if with_targets else None        

            index = round * chunk
            rest = min(inter_length - index, chunk)

            for recv_rank in range(client_size):
                if recv_rank == rank:
                    continue
                output_prg = mod_range(
                    shprg.genRandom(seed1s_s[recv_rank][rank][index:index + rest]),
                    p).astype(np.int64)
                data_to_client[:rest] += output_prg[:, :sub_features]
                if with_targets:
                    targets_to_client[:rest] += output_prg[:, sub_features:
                                                            sub_features +
                                                            target_length]
            
            temp_prg_dataset[rank].data[index:index + rest] = data_to_client[:rest]
            if with_targets:
                temp_prg_dataset[rank].targets[index:index + rest]=targets_to_client[:rest].ravel() if target_length == 1 else targets_to_client[:rest]
            return

        with ThreadPoolExecutor(max_workers=client_size +
                                client_size) as executor:
            sharerecv_args = [(rank,round)
                         for round in range(round_examples)
                         for rank in range(client_size)]
            tgt_prg_cal_args = [(rank,round)
                         for round in range(round_inter)
                         for rank in range(client_size)]
            executor.map(tgt_prg_cal_thread, tgt_prg_cal_args)
            executor.map(sharerecv_thread, sharerecv_args)
        # print(temp_prg_dataset[0].data[0,0])
        # print(temp_dataset[0].data[2,0])
        # print(temp_dataset[0].data[2,0]+temp_prg_dataset[0].data[0,0])
    else:
        with_targets = node.src_dataset.with_targets

        # for round in range(round_examples):
        def sharesend_thread(args):
            data_to_server = np.empty((chunk, sub_features), dtype=np.int64)
            targets_to_server = np.empty(
                (chunk, target_length), dtype=np.int64) if with_targets else None

            round = args
            index = round * chunk
            rest = min(sub_examples - index, chunk)
            for j in range(rest):
                perm_index = pre_permute[index + j]
                data_to_server[j] = encoder.encode(
                    node.src_dataset.data[perm_index])
                if with_targets:
                    targets_to_server[j] = encoder.encode(
                        node.src_dataset.targets[perm_index].reshape(
                            (1, target_length)))
            for k in range(client_size):
                if k == client_rank:
                    continue
                output_prg = mod_range(
                    shprg.genRandom(seeds[k][index:index + rest]),
                    p).astype(np.int64)
                data_to_server[:rest] -= output_prg[:, :sub_features]
                if with_targets:
                    targets_to_server[:rest] -= output_prg[:, sub_features:
                                                           sub_features +
                                                           target_length]
            if target_length == 1:
                node.send(
                    (data_to_server[:rest], targets_to_server[:rest].ravel()
                     if with_targets else None),
                    dest=server_rank,
                    tag=round)
                return
            node.send((data_to_server[:rest],
                       targets_to_server[:rest] if with_targets else None),
                      dest=server_rank,
                      tag=round)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            sharesend_args = [(round)
                         for round in range(round_examples)]
            executor.map(sharesend_thread, sharesend_args)

    timer.set_time_point("dset_share")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))

    # tgt dataset server send

    if is_server:
        def tgt_send_thread(args):
            # for rank in range(client_size):
            rank,round = args
            with_targets = (rank == targets_rank)
            # print(round)

            data_to_client = np.empty((chunk, sub_features), dtype=np.int64)
            targets_to_client = np.empty(
                (chunk,
                 target_length), dtype=np.int64) if with_targets else None
            
            index = round * chunk
            rest = min(inter_length - index, chunk)

            for j in range(rest):
                perm_index = permutes[rank][index + j]
                data_to_client[j] = temp_dataset[rank].data[perm_index]
                if with_targets:
                    targets_to_client[j] = temp_dataset[rank].targets[
                        perm_index].reshape((1, target_length))
            # if rank == 0:
            #     print(data_to_client[0,0])
            #     print(temp_prg_dataset[rank].data[0,0])
            #     print(data_to_client[0,0]+temp_prg_dataset[rank].data[0,0])

            data_to_client[:rest] += temp_prg_dataset[rank].data[
                index:index + rest]
            if with_targets:
                targets_to_client[:rest] += temp_prg_dataset[
                    rank].targets[index:index +
                                    rest].reshape((rest, target_length))
            # if rank == 0:
            #     print(data_to_client[0,0])
            if target_length == 1:
                node.send((data_to_client[:rest],
                            targets_to_client[:rest].ravel()
                            if with_targets else None),
                            dest=rank,
                            tag=round)
                return
            node.send((data_to_client[:rest],
                        targets_to_client[:rest] if with_targets else None),
                        dest=rank,
                        tag=round)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tgt_send_args = [(rank,round)
                         for round in range(round_inter)
                         for rank in range(client_size)]
            executor.map(tgt_send_thread, tgt_send_args)
    else:
        node.tgt_dataset.resize(data_shape = (inter_length,features), targets_shape = (inter_length,) if target_length == 1 else
                    (inter_length, target_length))
        def tgt_recv_thread(args):
            round, rank = args

            index = round * chunk
            rest = min(inter_length - index, chunk)
            
            if client_rank == rank:
                recv = node.recv(source=server_rank, tag=round)
                node.tgt_dataset.data[index:index + rest, rank * sub_features:(rank + 1) *
                        sub_features] = recv[0]
                if rank == targets_rank:
                    node.tgt_dataset.targets[index:index + rest] = recv[1]
                return
            output_prg = mod_range(
                shprg.genRandom(seed2s[rank][index:index +
                                                rest]), p).astype(np.int64)

            node.tgt_dataset.data[index:index + rest, rank * sub_features:(rank + 1) *
                    sub_features] = output_prg[:, :sub_features]
            if rank == targets_rank:
                node.tgt_dataset.targets[index:index + rest] = output_prg[:, sub_features:sub_features +                                             target_length].ravel() if target_length == 1 else output_prg[:, sub_features:sub_features +                                             target_length]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tgt_recv_args = [(round, rank)
                         for round in range(round_inter)
                         for rank in range(client_size)]
            executor.map(tgt_recv_thread, tgt_recv_args)

    timer.set_time_point("tgt_final ")
    print("{}: Rank {} - send: {:.4f} MB, recv: {:.4f} MB".format(
        timer.currentlabel, global_rank, node.getTotalDataSent(),
        node.getTotalDataRecv()))
    print("intersection size:{}".format(inter_length))
    print(timer)

    file.close()


if __name__ == "__main__":
    profiler = LineProfiler()
    profiler.add_function(main)
    profiler.run('main()')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arguments = sys.argv[1:]
    examples = int(arguments[0])
    features = int(arguments[1])

    output_filename = f'./data/lprof/poc2_profile_{examples}_{features}_rank_{rank}.txt'
    
    with open(output_filename, 'w') as output_file:
        profiler.print_stats(stream=output_file)
