from os import system as run

from matplotlib import pyplot as plt

lens = [1e1, 1e3, 1e7]
logs = {length: [[], []] for length in lens}

for length in lens:

    length = int(length)

    preprompt = f"python utils/create_data_for_task_2.py --length {length} --minv 10 --maxv 1000"
    run(preprompt)
    for proc_num in range(3, 13):

        filename = f"data/stat_{proc_num}_{length}"

        prompt = f"mpiexec -n {proc_num} python sources/task_2.py --filename data/array.in --length {length} --no-output > {filename}"
        run(prompt)

        with open(filename, "r") as file:
            lag = float(file.readline())

        logs[length][0].append(proc_num)
        logs[length][1].append(lag)

plt.xlabel("proc nums")
plt.ylabel("exec time")
plt.grid()
for length in lens:
    x, y = logs[length]
    plt.plot(x, y, label=str(length))
plt.legend(loc="upper right")

plt.savefig("plots/result_1.png")
plt.show()
