from cst.profiler.data import RunProfileData
from cst.memory import get_memory_curve

path = './logs/enformer/eagertrain_False-bz8-V100-bagg_False-mb_2/dgx2-4.hpc.engr.oregonstate.edu_315437.1740595825699720383.pt.trace.json'
name = 'enformer'


def find_node_by_name(node, name, results): 
    """
    Find a node by its name in the tree
    """
    if name in node.name:
        results.append(node)
    for child in node.children:
        find_node_by_name(child, name, results)
    
    return results


def remove_multi_profiler_steps(root_node):
    """
    Remove all the nodes after the first ProfilerStep#2
    """
    for node in root_node.children:
        if node.name == 'nn.Module: ModelWrapper_0':
            root_node.children = [node]
            break

def remove_multi_aten_ops(root_node):
    """
    Remove the aten nodes' children
    """
    if root_node.name.startswith('aten::'):
        root_node.children = []
        return
    for node in root_node.children:
        remove_multi_aten_ops(node)
        
def memory_curve(profiler):
    """
    Get the memory curve from the profiler
    """
    time_metric = 'ms'
    memory_metric = 'MB'
    memory_events = profiler._memory_events()
    memory_snapshot = profiler.memory_snapshot
    memory_curve = get_memory_curve(time_metric=time_metric, memory_metric=memory_metric, profiler_start_ts=profiler.profiler_start_ts, memory_records=memory_snapshot.memory_records)
    
    return memory_curve

# memory_curve(profile)

def draw_memory_curve(memory_curve, device='GPU0'):
    """
    绘制指定 device 的 memory curve，假设 GPU0 的数据格式为 [allocated, reserved, time]
    """
    import matplotlib.pyplot as plt

    gpu_data = memory_curve.get('rows', {}).get(device)
    if gpu_data is None:
        raise ValueError(f"没有找到设备 {device} 的数据")

    # 从每条数据中提取 allocated、reserved 和 time
    times = [row[0] for row in gpu_data]
    allocated = [row[1] for row in gpu_data]
    reserved = [row[2] for row in gpu_data]
    
    print(times[-1]-times[0])

    plt.figure(figsize=(12, 6))
    plt.plot(times, allocated, label='Allocated')
    plt.plot(times, reserved, label='Reserved')
    plt.xlabel('Time (ms)')
    plt.ylabel('Memory (MB)')
    plt.title(f'Memory Curve for {device}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./logs/{device}_memory_curve.png')
    plt.close()
    print(f"已保存 {device} 的 memory curve 到 ./logs/{device}_memory_curve.png")



if __name__ == '__main__':
    trace_path, trace_json = RunProfileData._preprocess_file(path, './logs/cache')
    profile = RunProfileData.from_json('worker_0', 1, trace_json)

    if profile.tid2tree:
        first_thread_id = next(iter(profile.tid2tree))
        root_node = profile.tid2tree[first_thread_id]
        print(f"Using thread ID: {first_thread_id}")
    else:
        raise ValueError("No thread data found in the profile")
    
    results = find_node_by_name(root_node, 'ProfilerStep#', [])
    print(f"Found {len(results)} nodes with name containing 'ProfilerStep#'")

    draw_memory_curve(memory_curve(profile), device='GPU0')

    # remove_multi_profiler_steps(root_node)
    # remove_multi_aten_ops(root_node)
    # draw_tree(root_node, './logs/tree_removed.png')
    