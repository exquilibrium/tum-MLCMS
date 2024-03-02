import matplotlib.pyplot as plt
import numpy as np

def get_group_fracs(file_name):
    text = list(x.rstrip() for x in open(file_name).readlines())
    col_num = str.split(text[0], " ").index("groupId-PID5")
    counts = {}
    for x in range(1, len(text)):
        id = int(str.split(text[x], " ")[0])
        sim_time = float(str.split(text[x], " ")[1])
        group = int(str.split(text[x], " ")[col_num])

        if sim_time in counts:
            counts[sim_time][group] += 1
        else:
            counts[sim_time] = [0, 0, 0]
            counts[sim_time][group] = 1

    sim_times = list(x for x in counts)
    group_counts = list(counts[x] for x in counts)

    def to_frac(t):
        s = sum(t)
        r = tuple(x / s for x in t)
        return r

    group_fracs = np.asarray(list(to_frac(x) for x in group_counts))
    return group_fracs, sim_times