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


@click.command()
def main():
    pass


if __name__ == "__main__":
    main()
