# comm_manage.py

import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# duplicate an existing communicator
comm_dup = comm.Dup()
print('comm_dup:', comm_dup)
comm_dup.Free()

# get the group associated with comm
grp = comm.Get_group()
# produce a group by exclude ranks 0 in `grp`
grp_excl = grp.Excl([0])
comm_excl = comm.Create(grp_excl)
if rank != 0:
    print('rank of comm {} -> rank of comm_excl: {}'.format(rank, comm_excl.rank))

# split comm into two new communicators according to `color`
color = rank % 2
comm_split = comm.Split(color=color, key=rank)
print('size of comm_split: %d' % comm_split.size)
print('rank of comm %d -> rank of comm_split: %d' % (rank, comm_split.rank))

if rank == 0:
    print('MPI.IDENT:', MPI.IDENT)
    print('MPI.CONGRUENT:', MPI.CONGRUENT)
    print('MPI.SIMILAR:', MPI.SIMILAR)
    print('MPI.UNEQUAL:', MPI.UNEQUAL)

if rank != 0:
    print(MPI.Comm.Compare(comm, comm_excl))
