import random
from struct import pack

import click


@click.command()
@click.option("--filename", "-f", default="data/array.in")
@click.option("--length", "-l", default=1000)
@click.option("--minv", "--mi", default=-100)
@click.option("--maxv", "--ma", default=100)
def main(filename, length, minv, maxv):
    randarr = [random.randint(minv, maxv) for i in range(length)]
    # print(f"Generated array: {randarr}")
    with open(filename, "wb") as file:
        # for item in randarr:
        #     file.write(pack("i", item))
        file.write(pack("i" * len(randarr), *randarr))
    s = sum(randarr)
    # print(f"sum of array = {s}")
    filesum = f"{filename.split('.')[0]}.sum"
    with open(filesum, "w") as file:
        file.write(str(s))


if __name__ == "__main__":
    main()
