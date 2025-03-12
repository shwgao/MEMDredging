import matplotlib.pyplot as plt
import numpy as np
from utils import read_from_file


color_table = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#9467bd', '#8c564b', 
                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_data_twinx(memory_tables, throughput_tables, legends, save_name=""):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(right=0.85)

    ax2 = ax1.twinx()
    
    for i, _ in enumerate(legends):
        ax1.plot(throughput_tables[i][1][0], throughput_tables[i][0][1], marker='o', 
                 label=f"Throughput ({legends[i]})", linestyle='--', color=color_table[i])
        ax2.plot(memory_tables[i][1][0], memory_tables[i][0][1], marker='s', 
                 label=f"Memory ({legends[i]})", linestyle='-', color=color_table[i])

        # ax1.plot(throughput_tables[i][1][0], throughput_tables[i][0][1], marker='^', 
        #          label=f"Throughput ({legends[i]})", linestyle='--', color=color_table[i + len(legends)])
        # ax2.plot(memory_tables[i][1][0], memory_tables[i][0][1], marker='d', 
        #          label=f"Memory ({legends[i]})", linestyle='-', color=color_table[i + len(legends)])

    # Customize axes
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (samples/sec)")
    ax2.set_ylabel("Memory (GB)")
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(x)
    ax1.set_ylim(max(throughput_tables[0][0][1]) * 0.6, max(throughput_tables[0][0][1]) * 1.1)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='upper center', 
            #    bbox_to_anchor=(1.05, 0.5)
               )

    plt.title(f"Throughput and Memory vs Batch Size")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./logs/{save_name}.png")

file_name1 = 'logs/simmim/pytorch-RTX8000-eager-train_False-bz29-bagg_True-mb_4.txt'
file_name2 = 'logs/simmim/pytorch-RTX8000-eager-train_False-bz29-bagg_False-mb_4.txt'

memory_table1, throughput_table1, batch_sizes1, stream_nums1 = read_from_file(file_name1)
memory_table2, throughput_table2, batch_sizes2, stream_nums2 = read_from_file(file_name2)
memory_tables = [
    [memory_table1, [batch_sizes1]], 
    [memory_table2, [batch_sizes2]]
]
throughput_tables = [
    [throughput_table1, [batch_sizes1]],
    [throughput_table2, [batch_sizes2]] 
]

plot_data_twinx(memory_tables, throughput_tables, legends=['microbatch', 'original'], save_name="simmim/pytorch-RTX8000-compare")
