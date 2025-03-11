from cst.profiler.data import RunProfileData
from cst.visualizer import draw_tree
path = './logs/cosmoflow/eagertrain_True-bz64-V100-bagg_False-mb_2/dgx2-3.hpc.engr.oregonstate.edu_2601904.1740811420313151476.pt.trace.json'
name = 'cosmoflow'

trace_path, trace_json = RunProfileData._preprocess_file(path, './logs/cache')
profile = RunProfileData.from_json('worker_0', 1, trace_json)

if profile.tid2tree:
    first_thread_id = next(iter(profile.tid2tree))
    root_node = profile.tid2tree[first_thread_id]
    print(f"Using thread ID: {first_thread_id}")
else:
    raise ValueError("No thread data found in the profile")

# draw_tree(root_node, './logs/tree.png')

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

remove_multi_profiler_steps(root_node)
remove_multi_aten_ops(root_node)
draw_tree(root_node, './logs/tree_removed.png')