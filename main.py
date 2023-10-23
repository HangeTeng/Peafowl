import sys
from mpi4py import MPI
import numpy as np
import math
from secrets import token_bytes
from src.communicator.node import Node
from src.utils.h5dataset import HDF5Dataset
from src.utils.crypto.prf import PRF
from src.utils.crypto.shprg import SHPRG
from src.utils.encoder import FixedPointEncoder, mod_range

if __name__ == "__main__":
    # dataset
    examples = 6
    features = 60
    chunk = 100
    # sub_dataset
    nodes = 3
    sub_examples = 5
    sub_features = 20
    targets_rank = 0

    secret_key = "secret_key"

    # shprg
    n = 2
    m = sub_features + 1
    EQ = 128
    EP = 64
    q = 2**EQ
    p = 2**EP
    seedA = bytes(0x355678)

    precision_bits = 16
    encoder = FixedPointEncoder(precision_bits=precision_bits)

    shprg = SHPRG(input=n, output=m, EQ=EQ, EP=EP)

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

    #* initial node
    if is_server:
        node = Node(None, None, global_comm, client_comm)
        temp_dataset = []
        for i in range(client_size):
            temp_path = "./data/temp/SVM_{}_{}_{}-{}_temp.hdf5".format(
                examples, features, i, nodes)
            temp_dataset.append(
                HDF5Dataset.empty(file_path=temp_path,
                                  data_shape=(sub_features, ),
                                  targets_shape=(),
                                  dtype=np.int64))
    else:
        src_path = "./data/SVM_{}_{}_{}-{}.hdf5".format(
            examples, features, global_rank, nodes)
        src_dataset = HDF5Dataset(file_path=src_path)
        tgt_path = "./data/SVM_{}_{}_{}-{}_share.hdf5".format(
            examples, features, global_rank, nodes)
        tgt_dataset = HDF5Dataset.empty(file_path=tgt_path,
                                        data_shape=(features, ),
                                        targets_shape=(),
                                        dtype=np.int64)
        node = Node(src_dataset, tgt_dataset, global_comm, client_comm)
    # print("start testing...")

    #* encrypted ID
    if is_server:
        id_enc = None
    else:
        prf = PRF(secret_key=secret_key)
        id_enc = np.vectorize(prf.compute)(node.src_dataset.ids[...])
    id_enc_gather = node.gather(id_enc, server_rank)
    # print(id_enc_gather)

    #* server-aid PSI
    if is_server:
        permutes, permute_length = node.find_intersection_indices(
            id_enc_gather[:-1])
        # print(permutes)
        # print(permute_length)
    else:
        pass

    #* seeds generation
    if is_server:
        pass
    else:
        seeds = [(None if i == client_rank else np.array(
            [[k + j * 10 + i * 100 + client_rank * 1000 for k in range(n)]
             for j in range(sub_examples)]))
                 for i in range(client_size)]  #! test
        # seeds = [(None if i == client_rank else SHPRG.genMatrixAES128(seed=token_bytes(16),n=n,m=sub_examples,EQ=EQ) ) for i in range(client_size)]

    #* share
    print("share...")
    round_examples = math.ceil(sub_examples / chunk)
    if is_server:
        for i in range(round_examples):
            for j in range(client_size):
                recv = node.recv(source=j, tag=i)
                temp_dataset[j].add(data=recv[0], targets=recv[1])
        # test
        # for client_rank_ in range(client_size):
        #     shprg = SHPRG(input=n, output=m, EQ=EQ, EP=EP)
        #     seeds = [(None if i == client_rank_ else np.array(
        #     [[k + j * 10 + i * 100 + client_rank_ * 1000 for k in range(n)]
        #      for j in range(sub_examples)])) for i in range(client_size)]
        #     for i in range(sub_examples):
        #         data = temp_dataset[client_rank_].data[i]
        #         if i == 0 and client_rank_ == 0:  print(data)
        #         for k in range(client_size):
        #             if k == client_rank_:
        #                 pass
        #             else:
        #                 output_prg = shprg.genRandom(seeds[k][i])
        #                 # if i == 0 and client_rank_ == 0:  print(output_prg) #!test
        #                 data = data + output_prg[:sub_features]
        #         if i == 0 and client_rank_ == 0:
        #             print(p)
        #             print(mod_range(data,p=p))
        #             print(data)

    else:
        with_targets = node.src_dataset.with_targets

        data_to_server = np.empty((0, sub_features), dtype=np.int64)
        targets_to_server = np.empty(
            (0, ), dtype=np.int64) if with_targets else None

        for i in range(round_examples):
            for j in range(chunk):
                index = i * chunk + j
                if index >= sub_examples:
                    break
                data = encoder.encode(node.src_dataset.data[index])

                # if index == 0 and client_rank == 0: print(data) #!test

                target = encoder.encode(
                    node.src_dataset.targets[index].reshape(
                        (1, ))) if with_targets else None
                for k in range(client_size):
                    if k == client_rank:
                        pass
                    else:
                        output_prg = shprg.genRandom(seeds[k][index])

                        # if index == 0 and client_rank == 0: print(output_prg) #!test

                        data = data - output_prg[:sub_features]

                        # if index == 0 and client_rank == 0: print(data) #!test

                        target = (target -
                                  output_prg[sub_features:sub_features +
                                             1]) if with_targets else None
                data = mod_range(data, p).astype(np.int64).reshape(
                    (1, sub_features))
                # if index == 0 and client_rank == 0: print(data) #!test
                target = (mod_range(target, p).astype(np.int64)).reshape(
                    (1, )) if with_targets else None
                # print(data_to_server.shape)
                # print(data.shape)
                data_to_server = np.concatenate((data_to_server, data), axis=0)
                if with_targets:
                    targets_to_server = np.concatenate(
                        (targets_to_server, target), axis=0)
            node.send((data_to_server, targets_to_server),
                      dest=server_rank,
                      tag=i)
            data_to_server = np.empty((0, sub_features), dtype=np.int64)
            targets_to_server = np.empty(
                (0, ), dtype=np.int64) if with_targets else None

    # seeds share
    print("seeds share...")
    if is_server:
        pass
    else:
        # print(seeds)
        seeds_exchanged = node.alltoall(seeds, in_clients=True)
        # print(seeds_exchanged)

    # Share Traslation
    print("Share Traslation...")
    if is_server:
        all_deltas = []
        for i in range(client_size):
            deltas = []
            for j in range(client_size):
                if i == j:
                    deltas.append(None)
                    continue
                delta = np.empty((sub_examples, n),dtype=object)
                for k in range(n):
                    _delta = node.STsend(size=sub_examples,
                                    permute=permutes[j],
                                    recver=i,
                                    tag=j+k*100)
                    delta[:, k] = _delta
                deltas.append(delta)
            all_deltas.append(deltas)       
        # print(all_deltas[0][2])
    else:
        a_s = []
        b_s = []
        for j in range(client_size):
            if client_rank == j:
                a_s.append(None)
                b_s.append(None)
                continue
            a = np.empty((sub_examples, n),dtype=object)
            b = np.empty((sub_examples, n),dtype=object)
            for k in range(n):
                _a,_b = node.STrecv(size=sub_examples, sender=server_rank, tag=j+k*100)
                a[:,k] = _a
                b[:,k] = _b
            a_s.append(a)
            b_s.append(b)
        permutes = [[4, 0, 3, 1, 2], [4, 3, 0, 2, 1], [1, 3, 4, 0, 2]]
        # if global_rank == 0:
            # print((a_s[2][permutes[2]]-b_s[2])%(2**128))
        
    # permute and share
    print("permute and share...")
    if is_server:
        seeds_exchanged = None
    else:
        for i in range(client_size):
            if client_rank == i:
                continue
            seeds_exchanged[i] = (seeds_exchanged[i] - a_s[i]) % q
    seeds_share_gather = node.gather(seeds_exchanged, server_rank)

    if is_server:
        # print(seeds_share_gather)
        # print(all_deltas)
        for i in range(client_size):
            for j in range(client_size):
                if i == j:
                    continue
                seeds_share_gather[i][j] = (seeds_share_gather[i][j][permutes[j]] + all_deltas[i][j]) % q
        seed1s_s = seeds_share_gather
        # print(seed1s_s[0])
    else:
        seed2s = b_s
        # if global_rank == 0:
            # print(seed2s)
    

    if is_server:
        pass
    else:
        permute_length = None
    permute_length = global_comm.bcast(permute_length, root = server_rank) # to make into node
    # print(permute_length)
    
    # tgt dataset server send
    round_inter = math.ceil(permute_length / chunk)
    if is_server:
        for rank in range(client_size):
            with_targets = (rank == targets_rank)

            data_to_client = np.empty((0, sub_features), dtype=np.int64)
            targets_to_client = np.empty(
                (0, ), dtype=np.int64) if with_targets else None
            
            for i in range(round_inter):
                for j in range(chunk):
                    index = i * chunk + j
                    if index >= permute_length:
                        break
                    perm_index = permutes[rank][index]
                    

                    data = temp_dataset[rank].data[perm_index]
                    target = temp_dataset[rank].targets[perm_index].reshape((1, )) if with_targets else None
                    for k in range(client_size):
                        if k == rank:
                            continue
                        output_prg = shprg.genRandom(seed1s_s[k][rank][index])
                        data = data + output_prg[:sub_features]
                        target = (target +
                                  output_prg[sub_features:m]) if with_targets else None
                    data = mod_range(data, p).astype(np.int64).reshape(
                    (1, sub_features))
                    target = (mod_range(target, p).astype(np.int64)).reshape(
                    (1, )) if with_targets else None
                    data_to_client = np.concatenate((data_to_client, data), axis=0)
                    if with_targets:
                        targets_to_client = np.concatenate(
                            (targets_to_client, target), axis=0)
                node.send((data_to_client, targets_to_client),
                      dest=rank,
                      tag=i)
                data_to_client = np.empty((0, sub_features), dtype=np.int64)
                targets_to_client = np.empty(
                    (0, ), dtype=np.int64) if with_targets else None

    else:
        index = [0] * client_size
        for i in range(round_inter):
            recv = node.recv(source=server_rank, tag=i)
            data = np.empty((len(recv[0]),0), dtype=np.int64)
            targets = np.empty((0, ), dtype=np.int64)
            for j in range(client_size):
                if client_rank == j:
                    data = np.concatenate((data, recv[0]), axis=1)
                    targets = np.concatenate(
                            (targets, recv[1]), axis=0) if j == targets_rank else targets
                    # print(targets)
                    continue
                _data = np.empty((0, sub_features), dtype=np.int64)
                _targets = np.empty((0, ), dtype=np.int64)
                for k in range(len(recv[0])):
                    output_prg = shprg.genRandom(seed2s[j][index[j]])
                    output_prg = mod_range(output_prg, p).astype(np.int64).reshape(
                    (1, m))
                    # print(output_prg[:,:sub_features])
                    _data = np.concatenate((_data,output_prg[:,:sub_features]), axis=0)
                    _targets = np.concatenate((_targets,output_prg[0,sub_features:m]), axis=0) if j == targets_rank else None
                    index[j] += 1
                data = np.concatenate((data, _data), axis=1)
                if _targets is None:
                    pass
                else:
                    targets = np.concatenate((targets, _targets), axis=0)
                # print(targets)
            node.tgt_dataset.add(data=data,targets=targets)
            print(tgt_dataset.data[:].tolist())