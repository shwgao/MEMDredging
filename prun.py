import torch_tb_profiler
from x.cst.run import RunProfile
from x.cst.profiler.data import RunProfileData


# 加载数据（保持不变）
path = './logs/cosmoflow/eagertrain_True-bz64-V100-bagg_False-mb_2/dgx2-3.hpc.engr.oregonstate.edu_2601904.1740811420313151476.pt.trace.json'
name = 'cosmoflow'

trace_path, trace_json = RunProfileData._preprocess_file(path, './logs/cache')
profile = RunProfileData.from_json('worker_0', 1, trace_json)

runpro = RunProfile('worker_0', 1)
runpro.memory_snapshots = profile.memory_snapshot

if profile.tid2tree:
    first_thread_id = next(iter(profile.tid2tree))
    root_node = profile.tid2tree[first_thread_id]
    print(f"Using thread ID: {first_thread_id}")
    me = profile._memory_events()
    # print(f"Memory events: {me}")
else:
    raise ValueError("No thread data found in the profile")

# Ensure `_get_profile_for_request` and `request` are defined or remove this block if unnecessary.
# For now, replacing with a placeholder to avoid errors.
time_metric = 'ms'
memory_metric = 'MB'
memory_curve = runpro.get_memory_curve(time_metric=time_metric, memory_metric=memory_metric)
print(f"Memory curve: {memory_curve}")


def draw_curve(curve):
    import matplotlib.pyplot as plt
    plt.plot(curve)
    plt.xlabel('Time')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage Over Time')
    plt.savefig(f'{name}_memory_curve.png')