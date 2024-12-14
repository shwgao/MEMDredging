import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

def read_from_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        memory_table_flag = False
        throughput_table_flag = False
        memory_table = {}
        throughput_table = {}
        batch_size = []
        
        for line in lines:
            # skip the empty line
            if line.strip() == "":
                continue
            
            if "Memory Table" in line:
                memory_table_flag = True
                throughput_table_flag = False
                continue
            elif "Throughput Table" in line:
                memory_table_flag = False
                throughput_table_flag = True
                continue
            
            # skip the head line
            if line.startswith("Stream Num\\Batch Size"):
                batch_size = [int(value.strip()) for value in line.split("|")[1:]]
                continue
                
            if memory_table_flag:
                line = line.strip().split("|")
                memory_table[int(line[0].strip())] = [float(value.strip()) if value.strip() != "None" else None for value in line[1:]]
            elif throughput_table_flag:
                line = line.strip().split("|")
                throughput_table[int(line[0].strip())] = [float(value.strip()) if value.strip() != "None" else None for value in line[1:]]
        
        stream_nums = list(memory_table.keys())
        return memory_table, throughput_table, batch_size, stream_nums


def plot_data_twinx(memory_table, throughput_table, stream_nums, batch_sizes):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(right=0.85)
    # color_table = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "gray", "brown", "black"]
    color_table = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Prepare data for plotting
    x = np.arange(len(stream_nums))
    batch_labels = [f"Batch={b}" for b in batch_sizes]

    # Plot throughput
    for i, batch_size in enumerate(batch_sizes):
        throughput = [throughput_table[j][i] for j in stream_nums]
        ax1.plot(x, throughput, marker='o', label=f"Throughput ({batch_labels[i]})", linestyle='--', color=color_table[i])

    # Plot memory (on a secondary y-axis)
    ax2 = ax1.twinx()
    for i, batch_size in enumerate(batch_sizes):
        memory = [memory_table[j][i] for j in stream_nums]
        ax2.plot(x, memory, marker='s', label=f"Memory ({batch_labels[i]})", linestyle='-', color=color_table[i])

    # Customize axes
    ax1.set_xlabel("Stream Number")
    ax1.set_ylabel("Throughput (samples/sec)")
    ax2.set_ylabel("Memory (GB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stream_nums)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='center left', 
               bbox_to_anchor=(1.05, 0.5))

    plt.title("Throughput and Memory vs Stream Number")
    plt.grid()
    plt.tight_layout()
    plt.savefig("./logs/batch_profile.png")

def plot_data_separate(table, stream_nums, batch_sizes, title=""):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(right=0.85)
    # color_table = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "gray", "brown", "black"]
    color_table = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Prepare data for plotting
    x = np.arange(len(stream_nums))
    batch_labels = [f"Batch={b}" for b in batch_sizes]

    # Plot throughput
    for i, batch_size in enumerate(batch_sizes):
        line = [table[j][i] for j in stream_nums]
        ax1.plot(x, line, marker='o', label=f"Throughput ({batch_labels[i]})", linestyle='-', color=color_table[i])

    # Customize axes
    ax1.set_xlabel("Stream Number")
    y_label = "Throughput (samples/sec)" if title == "Throughput" else "Memory (GB)"
    ax1.set_ylabel(y_label)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stream_nums)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.title(f"{title} vs Stream Number")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./logs/{title}.png")


if __name__ == "__main__":
    # Plot the data
    memory_table, throughput_table, batch_sizes, stream_nums = read_from_file("./logs/batch_profile.txt")
    # plot_data_twinx(memory_table, throughput_table, stream_nums, batch_sizes)
    plot_data_separate(memory_table, stream_nums, batch_sizes, "Memory")
    plot_data_separate(throughput_table, stream_nums, batch_sizes, "Throughput")
