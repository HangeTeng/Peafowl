# comm_group.py

import numpy as np
from mpi4py import MPI


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # get the group associated with comm
    grp = comm.Get_group()
    print('Group rank %d, group size: %d' % (grp.rank, grp.size))

    # create a new group by duplicate `grp`
    grp_new = grp.Dup()
    print('The duplicated new group:', grp_new)
    grp_new.Free()

    print('MPI.UNDEFINED:', MPI.UNDEFINED)

    # produce a group by include ranks 0, 1, 3 in `grp`
    grp_incl = grp.Incl([0, 1, 3])
    print('rank %d in grp -> rank %d in grp_incl' % (rank, grp_incl.rank))
    # produce a group by exclude ranks 1, 2 in `grp`
    grp_excl = grp.Excl([1, 2])
    print('rank %d in grp -> rank %d in grp_excl' % (rank, grp_excl.rank))

    grp_range_incl = grp.Range_incl([(0, 3, 2), (1, 2, 2)])
    print('rank %d in grp -> rank %d in grp_range_incl' % (rank, grp_range_incl.rank))
    grp_range_excl = grp.Range_excl([(0, 3, 2), (1, 2, 2)])
    print('rank %d in grp -> rank %d in grp_range_excl' % (rank, grp_range_excl.rank))

    # produce a group by combining `grp_incl` and `grp_excl`
    grp_union = MPI.Group.Union(grp_incl, grp_excl)
    print('Size of grp_union: %d' % grp_union.size)
    # produce a group as the intersection of `grp_incl` and `grp_excl`
    grp_intersect = MPI.Group.Intersect(grp_incl, grp_excl)
    print('Size of grp_intersect: %d' % grp_intersect.size)
    # grp_intersection = MPI.Group.Intersection(grp_incl, grp_excl)
    # print 'Size of grp_intersection: %d' % grp_intersection.size
    # produce a group from the difference of `grp_incl` and `grp_excl`
    grp_diff = MPI.Group.Difference(grp_incl, grp_excl)
    print('Size of grp_diff: %d' % grp_diff.size)

    # translate the ranks of processes in `grp_incl` to those in `grp_excl`
    print('translate:', MPI.Group.Translate_ranks(grp_incl, [0, 1, 2], grp_excl))

    print('MPI.IDENT:', MPI.IDENT)
    print('MPI.SIMILAR:', MPI.SIMILAR)
    print('MPI.UNEQUAL:', MPI.UNEQUAL)

    # compare `grp_incl` and `grp_incl`
    print(MPI.Group.Compare(grp_incl, grp_incl))
    # compare `grp_incl` and `grp_excl`
    print(MPI.Group.Compare(grp_incl, grp_excl))
