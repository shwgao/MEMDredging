from cst.profiler.data import RunProfileData
from cst.memory import get_memory_curve
import matplotlib.pyplot as plt


path = './logs/enformer/eagertrain_False-bz8-V100-bagg_False-mb_2/dgx2-4.hpc.engr.oregonstate.edu_315437.1740595825699720383.pt.trace.json'
path = './logs/enformer/eagertrain_True-bz2-V100-bagg_False-mb_4-check_False/tensorboard.pt.trace.json/dgx2-2.hpc.engr.oregonstate.edu_3726680.1742342040000112971.pt.trace.json'
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
    memory_snapshot = profiler.memory_snapshot
    memory_curve = get_memory_curve(time_metric=time_metric, memory_metric=memory_metric, profiler_start_ts=profiler.profiler_start_ts, memory_records=memory_snapshot.memory_records)
    
    return memory_curve

# memory_curve(profile)

def draw_memory_curve(memory_curve, device='GPU0'):
    """
    绘制指定 device 的 memory curve，数据格式为 [allocated, reserved, time]
    """

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
    plt.title(f'Memory Curve for Enformer with batch size 8(Inference)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./logs/Enformer_{device}_memory_curve.png')
    plt.close()
    print(f"已保存 {device} 的 memory curve 到 ./logs/Enformer_{device}_memory_curve.png")

def draw_histogram(event_types):
    """
    绘制事件类型的直方图
    """

    # 提取事件类型和数量
    event_names = list(event_types.keys())
    event_counts = list(event_types.values())

    # 绘制直方图
    plt.figure(figsize=(12, 6))
    plt.bar(event_names, event_counts)
    plt.xlabel('Event Types')
    plt.ylabel('Counts')
    plt.title('Event Type Distribution of Enformer(training)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./logs/event_type_distribution.png')
    plt.close()
    print("已保存事件类型分布图到 ./logs/event_type_distribution.png")
    

def hisgramm_event_types(profile):
    """
    统计事件类型的分布
    """
    event_types = {}
    for event in profile.clean_events:
        event_type = event.type
        if event_type not in event_types:
            event_types[event_type] = 0
        event_types[event_type] += 1
    
    return event_types


def event_distribution(trace_json):
    """
    统计事件类型的分布
    
    {'X': 94322, 'f': 12716, 'i': 13994, 's': 10180, 'M': 60}
    {'cpu_op': 43665, 'fwdbwd': 7344, 'cpu_instant_event': 13992, 'user_annotation': 9, 
    'python_function': 35089, 'kernel': 5751, 'ac2g': 15552, 'gpu_memset': 693, 'cuda_runtime': 8970, 
    'gpu_memcpy': 105, 'cuda_driver': 33, 'gpu_user_annotation': 6, None: 62, 'Trace': 1}
    """
    events = trace_json['traceEvents']
    event_types = {}
    event_categories = {}
    for event in events:
        event_type = event.get('ph')
        if event_type not in event_types:
            event_types[event_type] = 0
        event_types[event_type] += 1
        
        event_category = event.get('cat')
        if event_category not in event_categories:
            event_categories[event_category] = 0
        event_categories[event_category] += 1
    
    return event_types, event_categories


if __name__ == '__main__':
    trace_path, trace_json = RunProfileData._preprocess_file(path, './logs/cache')
    event_types, event_categories = event_distribution(trace_json)
    print(event_types)
    print(event_categories)
    
    # profile = RunProfileData.from_json('worker_0', 1, trace_json)
    profile = RunProfileData.parse('worker_0', 1, path, './logs/cache')
    
    profile.data_clean_tree()
    profile.data_clean()
    profile.events = profile.clean_events
    profile.process()

    if profile.tid2tree:
        first_thread_id = next(iter(profile.tid2tree))
        root_node = profile.tid2tree[first_thread_id]
        print(f"Using thread ID: {first_thread_id}")
    else:
        raise ValueError("No thread data found in the profile")
    
    results = find_node_by_name(root_node, 'ProfilerStep#', [])
    print(f"Found {len(results)} nodes with name containing 'ProfilerStep#'")

    # draw_memory_curve(memory_curve(profile), device='GPU0')
    event_types = hisgramm_event_types(profile)
    draw_histogram(event_types)

    # remove_multi_profiler_steps(root_node)
    # remove_multi_aten_ops(root_node)
    # draw_tree(root_node, './logs/tree_removed.png')
    