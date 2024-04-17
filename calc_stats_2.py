from os import system as run

from matplotlib import pyplot as plt

lens = [1e1, 1e3, 1e4, 1e5]
logs = {length: [[], []] for length in lens}

for length in lens:

    length = int(length)
    for proc_num in range(2, 13):
        print(f"Processing len={length}, proc_num={proc_num}...", end='\r')

        filename = f"data/stat_task3_{proc_num}_{length}"

        prompt = (
            f"mpiexec -n {proc_num} python task_3.py -x {length} --no-output > {filename}"
        )
        run(prompt)

        with open(filename, "r") as file:
            lag = float(file.readline())

        logs[length][0].append(proc_num)
        logs[length][1].append(lag)

print()
print()

for length in lens:
    # if length == int(1e7):
    plt.plot(*logs[length])
    # break
plt.show()
