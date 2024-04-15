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


def get_chunk_params(i, k, length):
    q = length % k
    p = length // k
    chunk_start = (p + 1) * min(i, q) + p * max(0, i - q)
    chunk_size = p
    if i < q:
        chunk_size += 1
    chunk_start *= sizeof
    return chunk_start, chunk_size


def calc_sum(comm, rank, size, length, file, no_output):

    status = MPI.Status()

    chunk_start, chunk_size = get_chunk_params(rank, size, length)
    buf = empty(chunk_size, dtype=np.int32)

    file.Read_at_all(chunk_start, [buf, chunk_size, MPI.INT32_T])
    if not no_output:
        print(f"{rank}th processes buf: {buf}")

    result = array(sum(buf), dtype=np.int64)
    if in_master(rank):
        subsums = result
        chunk = empty(1, dtype=np.int64)
        for idx in range(1, size):
            comm.Recv([chunk, 1, MPI.INT64_T], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            subsums += chunk[0]
        global_sum = subsums
        if not no_output:
            print(f"Data collected, sum = {global_sum}")
    else:
        comm.Send([result, 1, MPI.INT64_T], dest=master, tag=0)


@click.command()
@click.option("--filename", "-v", default="data/array.txt", help="Path to txt file with array")
@click.option("--length", "-l", default=10, help="Size of the array")
@click.option("--no-output", is_flag=True, help="Do you want to hide output? Please set it up!")
def cli(filename, length, no_output):

    start = time()

    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    # print(f"{rank}, {size}")

    file = MPI.File.Open(comm, filename, amode=MPI.MODE_WRONLY)

    calc_sum(comm, rank, size, length, file, no_output)
    file.Close()

    if in_master(rank):
        lag = time() - start
        print(lag)


if __name__ == "__main__":
    cli()
