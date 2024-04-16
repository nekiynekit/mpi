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


def calc_sum(comm, rank, size, file, no_output, rcounts, disples):

    # get sizes and starts
    chunk_start, chunk_size = get_chunk_params(rank, rcounts, disples)
    buf = empty(chunk_size, dtype=np.int32)

    # reading a file
    file.Read_at_all(chunk_start * sizeof, [buf, chunk_size, MPI.INT32_T])
    if not no_output:
        print(f"{chunk_start=}, {chunk_size=}")
        print(f"{rank}th processes buf: {buf}")

    # get subsum
    result = array(sum(buf), dtype=np.int64)

    # send subsum to master process
    rcounts = array([min(1, i) for i in range(size)], dtype=np.int32)
    disples[1:] = [i for i in range(size - 1)]
    if in_master(rank):
        terms = empty(size - 1, dtype=np.int64)
    else:
        terms = None
    comm.Gatherv([result, 1, MPI.INT64_T], [terms, rcounts, disples, MPI.INT64_T], root=0)

    # printing result
    if in_master(rank):
        global_sum = sum(terms)
        if not no_output:
            print(f"Data collected, sum = {global_sum}")


@click.command()
@click.option("--filename", "-v", default="data/array.txt", help="Path to txt file with array")
@click.option("--length", "-l", default=10, help="Size of the array")
@click.option("--no-output", is_flag=True, help="Do you want to hide output? Please set it up!")
def cli(filename, length, no_output):

    start = time()

    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()
    rcounts, disples = precalc_rcount_disples(comm, rank, size, length)

    file = MPI.File.Open(comm, filename, amode=MPI.MODE_WRONLY)

    calc_sum(comm, rank, size, file, no_output, rcounts, disples)
    file.Close()

    if in_master(rank):
        lag = time() - start
        print(lag)


if __name__ == "__main__":
    cli()
