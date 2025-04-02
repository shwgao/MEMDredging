from cst.profiler.data import RunProfileData
from cst.memory import get_memory_curve
from cst.profiler.module_op import get_module_tree
import matplotlib.pyplot as plt


path1 = './logs/enformer/eagertrain_False-bz8-V100-bagg_False-mb_2/dgx2-4.hpc.engr.oregonstate.edu_315437.1740595825699720383.pt.trace.json'
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
    memory_snapshot = profiler.clean_memory_snapshot
    memory_curve = get_memory_curve(time_metric=time_metric, memory_metric=memory_metric, 
                                    profiler_start_ts=profiler.last_step_ts, 
                                    memory_records=memory_snapshot.memory_records)
    
    return memory_curve

# memory_curve(profile)

def draw_memory_curve(memory_curve, device='GPU0', name=''):
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
    plt.title(f'Memory Curve for Enformer with batch size 8({name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./logs/Enformer_{device}_memory_curve_{name}.png')
    plt.close()
    print(f"已保存 {device} 的 memory curve 到 ./logs/Enformer_{device}_memory_curve_{name}.png")

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
    plt.savefig(f'./logs/event_type_distribution_{name}.png')
    plt.close()
    print(f"已保存事件类型分布图到 ./logs/event_type_distribution_{name}.png")
    

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


def get_profile_from_json(path):
    trace_path, trace_json = RunProfileData._preprocess_file(path, './logs/cache')
    event_types, event_categories = event_distribution(trace_json)
    print(event_types)
    print(event_categories)
    
    # profile = RunProfileData.from_json('worker_0', 1, trace_json)
    profile = RunProfileData.parse('worker_0', 1, path, './logs/cache')
    
    profile.data_clean_tree()
    profile.clean_memory_record()
    
    module_tree = get_module_tree(profile.clean_tid2tree)
    return profile, module_tree


def count_childern_num(root_node):
    """
    计算每个节点的子节点数量
    """
    if root_node.children is None:
        return 0
    
    for node in root_node.children:
        if not hasattr(node, 'childern_num'):
            node.childern_num = 0
        node.childern_num += count_childern_num(node)
        
    return node.childern_num
        

def memory_property_set(root_node, memory_events):
    # convert memory B to MB
    for event in memory_events:
        event._total_allocated = event.total_allocated / 1024 / 1024
        event._total_reserved = event.total_reserved / 1024 / 1024
        event._bytes = event.bytes / 1024 / 1024
    
    # set attributes for each node
    def set_attributes(node):
        node.peak_memory = 0
        node.allocated_span = 0
        node.released_span = 0
        for child in node.children:
            set_attributes(child)
        
    def traverse_tree(node, event):
        if node.start_time <= event.ts and node.end_time >= event.ts:
            node.peak_memory = max(node.peak_memory, event._total_allocated)
            if event._bytes > 0:
                node.allocated_span += event._bytes
            else:
                node.released_span += -(event._bytes)
        else:
            return
                
        for child in node.children:
            traverse_tree(child, event)

    set_attributes(root_node)
    
    for event in memory_events:
        traverse_tree(root_node, event)


def cost_tree(root_node):
    """
    计算每棵树的 cost
    """
    for node in root_node.children:
        node.cost = node.peak_memory * (node.end_time - node.start_time)
        

def tree_sub_nodes_num(root_node):
    """为每棵树的每个节点添加子树节点数量属性（包含节点本身）"""
    def set_sub_nodes_num(node):
        node.sub_nodes_num = 1  # 初始化为1，计入节点本身
        
        if not node.children:
            return 1
        
        for child in node.children:
            node.sub_nodes_num += set_sub_nodes_num(child)
        
        return node.sub_nodes_num
    
    set_sub_nodes_num(root_node)

if __name__ == '__main__':
    # profile_train, module_tree_train = get_profile_from_json(path)
    profile_inference, module_tree_inference = get_profile_from_json(path1)
    
    clean_memory_events = profile_inference.clean_memory_events
    memory_property_set(module_tree_inference[0], clean_memory_events)
    
    draw_memory_curve(memory_curve(profile_inference), device='GPU0', name='inference')
    event_types = hisgramm_event_types(profile_inference)
    draw_histogram(event_types)

    # remove_multi_profiler_steps(root_node)
    # remove_multi_aten_ops(root_node)
    # draw_tree(root_node, './logs/tree_removed.png')
    