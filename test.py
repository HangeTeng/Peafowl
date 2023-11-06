# print((75084707279165425167553911412330502867+265197659641773038295820696019437708809)%(2**128))
# print((228165717330848606164949012111148510838+112116649590089857298425595320619701638)%(2**128))
# print(-3490860109668371091+6109467841530786198)
# print((2618607731862476205+7914068170923539857+7914068170923539857)%2**64)
# print((3471918219412512322+1853755705742518824+1853755705742518824))


# def has_duplicates(nums):
#     seen = set()
#     for num in nums:
#         if num in seen:
#             return True
#         seen.add(num)
#     return False

# numbers = [0, 31, 62, 93, 124, 73, 104, 135, 166, 197, 146, 177, 208, 239, 270, 219, 250, 281, 312, 343, 292, 323, 354, 385, 416, 109, 140, 171, 202, 233, 182, 213, 244, 275, 306, 255, 286, 317, 348, 379, 328, 359, 390, 421, 452, 401, 432, 463, 494, 525, 218, 249, 280, 311, 342, 291, 322, 353, 384, 415, 364, 395, 426, 457, 488, 437, 468, 499, 530, 561, 510, 541, 572, 603, 634, 327, 358, 389, 420, 451, 400, 431, 462, 493, 524, 473, 504, 535, 566, 597, 546, 577, 608, 639, 670, 619, 650, 681, 712, 743, 436, 467, 498, 529, 560, 509, 540, 571, 602, 633, 582, 613, 644, 675, 706, 655, 686, 717, 748, 779, 728, 759, 790, 821, 852, 545, 576, 607, 638, 669, 618, 649, 680, 711, 742, 691, 722, 753, 784, 815, 764, 795, 826, 857, 888, 837, 868, 899, 930, 961, 654, 685, 716, 747, 778, 727, 758, 789, 820, 851, 800, 831, 862, 893, 924, 873, 904, 935, 966, 997, 946, 977, 1008, 1039, 1070, 763, 794, 825, 856, 887, 836, 867, 898, 929, 960, 909, 940, 971, 1002, 1033, 982, 1013, 1044, 1075, 1106, 1055, 1086, 1117, 1148, 1179]

# if has_duplicates(numbers):
#     print("有重复的数字")
# else:
#     print("没有重复的数字")

# """
# -
# -777068238512547296
# 5476144912792647449


# """
from mpi4py import MPI
from src.communicator.STComm import Sender, Receiver


def STinit(is_server,STSenders,STRecver,size=None, permutes=None, p=1<<128, ios_threads = 4):
    if is_server:
        if size is None or permutes is None:
            raise ValueError("Invalid: missing size or permute")
        for i in range(len(permutes)):
            STSenders[i] = Sender(size=size, permute=permutes[i], p=p, ios_threads = ios_threads)
    else:
        STRecver[0] = Receiver(ios_threads = ios_threads)

def STsend( size,
            dset_rank,
            STSenders,
            recver=None,
            tag=0,
            p=1 << 128,
            Sip="127.0.0.1",
            port = 40280,
            ot_type=1,
            num_threads=2,
            port_mode = True,):
    comm = MPI.COMM_WORLD
    sessionHint = str(tag) if recver is None else (str(
        comm.Get_rank()) + "_" + str(recver) + "_" + str(tag))
    # result = self.STSender.run(size=size,
    if port_mode:
        all_ip = Sip +":"+str(port + tag)
        STSender = STSenders[dset_rank]
    else:
        all_ip = Sip +":"+str(port+ dset_rank )
        STSender = STSenders[dset_rank]
    result = STSender.run(size=size,
                    sessionHint=sessionHint,
                    p=p,
                    Sip=all_ip,
                    ot_type=ot_type,
                    num_threads=num_threads)
    return result

def STrecv(            size,
            dset_rank,
            STRecver,
            sender=None,
            in_clients=False,
            tag=0,
            p=1 << 128,
            Sip="127.0.0.1",
            port = 40280,
            ot_type=1,
            num_threads=2,
            port_mode = True,):
    comm = MPI.COMM_WORLD
    sessionHint = str(tag) if sender is None else (str(sender) + "_" + str(
        comm.Get_rank()) + "_" + str(tag))
    
    if port_mode:
        all_ip = Sip +":"+str(port + tag)
        Recver = STRecver[0]
    else:
        all_ip = Sip +":"+str(port + dset_rank)
        Recver = STRecver[0]
    result = Recver.run(size=size,
                        sessionHint=sessionHint,
                        p=p,
                        Sip=all_ip,
                        ot_type=ot_type,
                        num_threads=num_threads)
    return result[0],result[1]

def STsend_thread(args, STSenders):
    all_deltas,sub_examples,port,num_threads, rank, dset_rank, input_dim = args
    if rank == dset_rank:
        return
    all_deltas[rank, dset_rank, :, input_dim] = STsend(
    # STsend(
        STSenders=STSenders,
        size=sub_examples,
        dset_rank=dset_rank,
        recver=rank,
        tag= rank * 31 + dset_rank * 73 + input_dim * 109,
        port = port,
        port_mode=False,
        num_threads = num_threads)
    
def STrecv_thread(args):
    sub_examples, client_rank,server_rank, port,num_threads, STRecvers,dset_rank, input_dim = args
    if client_rank == dset_rank:
        return
    # a_s[dset_rank, :,
        # input_dim], b_s[dset_rank, :, input_dim] = node.STrecv(
    STrecv( STRecver = STRecvers[0],
            size=sub_examples,
            dset_rank=dset_rank,
            sender=server_rank,
            tag= client_rank * 31 + dset_rank * 73 + input_dim * 109,
            port = port,
            port_mode=False,
            num_threads = num_threads)


def main():
    import numpy as np
    # shprg
    n = 2
    EQ = 128
    EP = 64
    q = 2**EQ
    p = 2**EP
    seedA = bytes(0x355678)

    # encoder
    precision_bits = 16

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
    max_worker = 10 * client_size
    server_max_worker = max_worker * client_size
    num_threads = 1
    is_server = False
    if global_rank == global_size - 1:
        is_server = True

    


    STSenders = [None] * (global_comm.Get_size() - 1)
    STRecvers = [None]



    n = 1
    sub_examples = 10000
    port = 30000

    import random

    if is_server:
        permutes = []
        for _ in range(client_size):
            random.seed(1)
            all_indices = list(range(sub_examples))
            random.shuffle(all_indices)
            permutes.append(all_indices)
        STinit(is_server,STSenders,STRecvers,size=sub_examples,permutes=permutes,p=q)
        # print(node.STSenders)
        # print(permutes)
    else:
        STinit(is_server,STSenders,STRecvers)
        pass

    
    import os
    os.environ['RDMAV_FORK_SAFE'] = '1'
    from concurrent.futures import ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=server_max_worker)

    from functools import partial
    
    if is_server:
        all_deltas = np.empty((client_size, client_size, sub_examples, n),
                                dtype=object)
            
        task_args = [(all_deltas,sub_examples,port,num_threads, rank, dset_rank, input_dim)
                        for input_dim in range(n)
                        for dset_rank in range(client_size)
                        for rank in range(client_size)]
        results = list(executor.map(partial(STsend_thread, STSenders=STSenders), task_args))
        print(all_deltas[0,1,0,:])

    else:
        a_s = np.empty((client_size, sub_examples, n), dtype=object)
        b_s = np.empty((client_size, sub_examples, n), dtype=object)


        # task_args = [(sub_examples, client_rank,server_rank, port,num_threads, STRecvers,dset_rank, input_dim)
        #                 for input_dim in range(n)
        #                 for dset_rank in range(client_size)
        #                 ]
        
        # results = list(executor.map(STrecv_thread, task_args))

from line_profiler import LineProfiler

if __name__ == "__main__":
    profiler = LineProfiler()
    profiler.add_function(main)
    profiler.run('main()')

    output_filename = f'./test.txt'

    with open(output_filename, 'w') as output_file:
        profiler.print_stats(stream=output_file)
    # output_filename = f'./data/lprof/poc2_profile_{examples}_{features}_rank_{rank}.cprof.txt'
    # cProfile.run("main()", output_filename, sort="cumulative")