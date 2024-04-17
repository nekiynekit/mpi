from math import cos, sin
from time import time

import click
import numpy as np
from mpi4py import MPI
from numpy import array, empty

master = 0
sizeof = 4  # number of bytes in int32


def in_master(rank):
    global master
    return master == rank


def precalc_rcount_disples(comm, rank, size, length):
    if rank == 0:
        ave, res = divmod(length, size - 1)
        rcounts = empty(size, dtype=np.int32)
        displs = empty(size, dtype=np.int32)

        rcounts[0] = 0
        displs[0] = 0

        for k in range(1, size):
            rcounts[k] = ave + 1 if k <= res else ave
            displs[k] = displs[k - 1] + rcounts[k - 1]

    else:
        rcounts = empty(size, dtype=np.int32)
        displs = empty(size, dtype=np.int32)

    comm.Bcast([rcounts, size, MPI.INT], root=0)
    comm.Bcast([displs, size, MPI.INT], root=0)

    return rcounts, displs


def get_chunk_params(i, rcounts, disples):
    chunk_size = rcounts[i]
    chunk_start = disples[i]
    return chunk_start, chunk_size


def f(x, y):
    return sin(x) + cos(y)


def calc_diff(comm, N, M, rank, rcounts, disples, no_output):

    chunk_start, chunk_size = get_chunk_params(rank, rcounts, disples)
    my_mat = empty((chunk_size, M), dtype=np.float32)
    my_diff = empty(chunk_size * M, dtype=np.float32)

    for i in range(chunk_size):
        for j in range(M):
            x, y = i + chunk_start, j
            my_mat[i][j] = f(x, y)
    for i in range(chunk_size):
        for j in range(M):
            my_diff[i * M + j] = my_mat[i][j] - my_mat[i][j - 1] if j > 0 else 0.0

    if in_master(rank):
        total_mat = empty(N * M, np.float32)
    else:
        total_mat = None

    def factor_x(container, x):
        return [item * x for item in container]

    rcounts, disples = factor_x(rcounts, M), factor_x(disples, M)
    comm.Gatherv([my_diff, chunk_size * M, MPI.FLOAT], [total_mat, rcounts, disples, MPI.FLOAT])

    if not no_output and in_master(rank):
        print(f"Differential[{N - 1}][{M - 1}]={total_mat[N * M - 1]}")


@click.command()
@click.option("--ox", "-x", default=100000, type=int)
@click.option("--oy", "-y", default=100, type=int)
@click.option("--no-output", is_flag=True, help="Do you want to hide output? Please set it up!")
def cli(ox, oy, no_output):

    start = time()

    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()
    rcounts, disples = precalc_rcount_disples(comm, rank, size, ox)

    calc_diff(comm, ox, oy, rank, rcounts, disples, no_output)

    if in_master(rank):
        lag = time() - start
        print(lag)


if __name__ == "__main__":
    cli()
