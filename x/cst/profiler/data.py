# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import gzip
import io as sysio
import json
import re
import tempfile
import copy
from json.decoder import JSONDecodeError
from typing import Dict, List, Optional

from .. import io, utils
from ..utils import href
from . import trace
from .communication import analyze_communication_nodes
from .event_parser import CommLibTypes, EventParser, ProfileRole
from .gpu_metrics_parser import GPUMetricsParser
from .kernel_parser import KernelParser
from .memory_parser import MemoryParser, MemorySnapshot
from .node import OperatorNode
from .op_agg import ModuleAggregator
from .overall_parser import OverallParser
from .tensor_cores_parser import TensorCoresParser
from .trace import BaseEvent, EventTypes, MemoryEvent

logger = utils.get_logger()


class RunProfileData:
    def __init__(self, worker: str, span: str, trace_json: Dict):
        self.worker = worker
        self.span = span

        # metadatas
        self.is_pytorch_lightning = trace_json.get('Framework', None) == 'pytorch-lightning'
        self.data_schema_version = trace_json.get('schemaVersion', None)
        self.distributed_info = trace_json.get('distributedInfo', None)
        self.device_props = trace_json.get('deviceProperties', None)

        self.profiler_start_ts = float('inf')
        self.events: List[BaseEvent] = []

        trace_body = trace_json['traceEvents']
        fwd_bwd_events = []
        fwdbwd_events = []
        ac2g_events = []
        for data in trace_body:
            if data.get('cat') == 'forward_backward':
                fwd_bwd_events.append(data)
            else:
                event = trace.create_event(data, self.is_pytorch_lightning)
                if event is not None:
                    self.profiler_start_ts = min(self.profiler_start_ts, event.ts)
                    self.events.append(event)
            
            if data.get('cat') == 'fwdbwd':
                fwdbwd_events.append(data)
            if data.get('cat') == 'ac2g':
                ac2g_events.append(data)

        self.events.sort(key=lambda e: e.ts)
        self.forward_backward_events = trace.create_association_events(fwd_bwd_events)
        self.fwdbwd_events = trace.create_association_events(fwdbwd_events)
        self.ac2g_events = trace.create_association_events(ac2g_events)
        self.trace_file_path: str = None

        # Event Parser results
        self.tid2tree: Dict[int, OperatorNode] = None
        self.pl_tid2tree: Dict[int, OperatorNode] = None
        self.used_devices = []
        self.use_dp: bool = False
        self.use_ddp: bool = False
        self.comm_lib = None
        self.has_runtime: bool = False
        self.has_kernel: bool = False
        self.has_communication: bool = False
        self.has_memcpy_or_memset: bool = False
        self.role_ranges = None
        self.steps_costs = None
        self.steps_names = None
        self.avg_costs = None

        # GPU parser
        self.gpu_metrics_parser: GPUMetricsParser = None

        # Operator aggregator
        self.op_list_groupby_name = None
        self.op_list_groupby_name_input = None
        self.stack_lists_group_by_name = None
        self.stack_lists_group_by_name_input = None
        self.kernel_list_groupby_name_op = None

        # Kernel and Tensor Core
        self.kernel_stat = None
        self.tc_ratio = None
        self.tc_eligible_ops_kernel_ratio = None
        self.tc_used_ratio = None  # If it's a pure CPU run, then this keeps as None.

        # Communicator
        self.comm_node_list = None
        self.comm_overlap_costs = None
        self.memory_snapshot: Optional[MemorySnapshot] = None

        # recommendation based on analysis result.
        self.recommendations = []
        
        # data_clean
        self.clean_root_node = None
        self.clean_events = []
        self.clean_tid2tree = {}
        self.clean_profiler_start_ts = -1
        self.clean_memory_snapshot: Optional[MemorySnapshot] = None
        self.steps = []
        self.last_step_name = None
        self.last_step_ts = float('inf')
        self.last_step_te = 0
        self.last_step_memory_events = []
        
    @staticmethod
    def parse(worker, span, path, cache_dir):
        trace_path, trace_json = RunProfileData._preprocess_file(path, cache_dir)

        profile = RunProfileData.from_json(worker, span, trace_json)
        profile.trace_file_path = trace_path
        return profile

    @staticmethod
    def from_json(worker, span, trace_json: Dict):
        profile = RunProfileData(worker, span, trace_json)
        with utils.timing('Data processing'):
            profile.process()
        profile.analyze()
        return profile

    @staticmethod
    def _preprocess_file(trace_path, cache_dir):
        if not io.exists(trace_path):
            raise FileNotFoundError(trace_path)

        data = io.read(trace_path)
        if trace_path.endswith('.gz'):
            data = gzip.decompress(data)

        json_reencode = False
        try:
            trace_json = json.loads(data)
        except JSONDecodeError as e:
            # Kineto may export json file with non-ascii code. before this is fixed, use a workaround
            # to handle JSONDecodeError, re-encode it and save to a temp file
            try:
                trace_json = json.loads(data, strict=False)
            except JSONDecodeError:
                with sysio.StringIO() as fout:
                    str_data = data.decode('utf-8')
                    # only replace the N/A without surrounding double quote
                    fout.write(re.sub(r'(?<!")N/A(?!")', "\"N/A\"", str_data))
                    trace_json = json.loads(fout.getvalue())
                    logger.warning('Get JSONDecodeError: %s, Re-encode it to temp file' % e.msg)
                    json_reencode = True

        # work-around to remove the 'Record Window End' events to avoid the huge end timestamp
        event_list = trace_json['traceEvents']
        end_index = None
        start_index = None
        for i in reversed(range(len(event_list))):
            if event_list[i]['name'] == 'Record Window End':
                end_index = i
            elif event_list[i]['name'].startswith('Iteration Start:'):
                start_index = i
            if start_index is not None and end_index is not None:
                break

        if start_index is not None and end_index is not None:
            dur = event_list[end_index]['ts'] - event_list[start_index]['ts']
            if dur > 24 * 3600 * 1000:
                del trace_json['traceEvents'][end_index]
                json_reencode = True

        if json_reencode:
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', dir=cache_dir, delete=False)
            fp.close()
            with gzip.open(fp.name, mode='wt') as fzip:
                fzip.write(json.dumps(trace_json))
            trace_path = fp.name

        return trace_path, trace_json

    def process(self):
        with utils.timing('EventParser.parse'):
            parser = EventParser()
            self.tid2tree, self.pl_tid2tree = parser.parse(self.events, self.forward_backward_events)
            self.steps = parser.steps            

        self.has_runtime = parser.has_runtime
        self.has_kernel = parser.has_kernel
        self.has_memcpy_or_memset = parser.has_memcpy_or_memset
        self.steps_names = parser.steps_names
        self.used_devices = sorted(list(parser.used_devices))
        self.use_dp = parser.use_dp
        self.use_ddp = parser.use_ddp
        self.role_ranges = parser.role_ranges

        self.comm_lib = parser.comm_lib
        self.has_communication = parser.has_communication
        self.comm_node_list = parser.comm_node_list

        # Starting aggregate
        logger.debug('ModuleAggregator')
        with utils.timing('ModuleAggregator aggegation'):
            module_aggregator = ModuleAggregator()
            module_aggregator.aggregate(self.tid2tree)
        self.op_list_groupby_name = module_aggregator.op_list_groupby_name
        self.op_list_groupby_name_input = module_aggregator.op_list_groupby_name_input
        self.stack_lists_group_by_name = module_aggregator.stack_lists_group_by_name
        self.stack_lists_group_by_name_input = module_aggregator.stack_lists_group_by_name_input
        self.kernel_list_groupby_name_op = module_aggregator.kernel_list_groupby_name_op

        logger.debug('OverallParser')
        with utils.timing('OverallParser aggegation'):
            overall_parser = OverallParser()
            overall_parser.aggregate(parser.steps, parser.role_ranges)
        self.avg_costs = overall_parser.avg_costs
        self.steps_costs = overall_parser.steps_costs
        self.comm_overlap_costs = overall_parser.communication_overlap

        logger.debug('GPUMetricsParser')
        self.gpu_metrics_parser = GPUMetricsParser.parse_events(
            self.events, parser.global_start_ts, parser.global_end_ts, parser.steps[0][0], parser.steps[-1][1])

        logger.debug('TensorCoresParser')
        tensorcores_parser = TensorCoresParser.parse_events(
            self.tid2tree, module_aggregator.ops, self.gpu_metrics_parser.gpu_ids)
        self.tc_eligible_ops_kernel_ratio = tensorcores_parser.tc_eligible_ops_kernel_ratio
        self.tc_ratio = tensorcores_parser.tc_ratio

        if self.has_kernel:
            logger.debug('KernelParser')
            with utils.timing('parse kernels'):
                kernel_parser = KernelParser()
                kernel_parser.parse_events(self.events)
            self.kernel_stat = kernel_parser.kernel_stat
            self.tc_used_ratio = kernel_parser.tc_used_ratio

        memory_events = self._memory_events()
        if memory_events:
            memory_parser = MemoryParser(memory_events)
            self.memory_snapshot = memory_parser.find_memory_nodes(self.tid2tree)

    def analyze(self):
        self.recommendations = []

        dataloader_ratio = self.avg_costs.costs[ProfileRole.DataLoader] / self.avg_costs.costs[ProfileRole.Total]
        if dataloader_ratio > 0.05:
            percentage = dataloader_ratio * 100
            url = 'https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading'
            self.recommendations.append(
                f'This run has high time cost on input data loading. {percentage:.1f}% of the step ' +
                "time is in DataLoader. You could try to set num_workers on DataLoader's construction " +
                f"and {href('enable multi-processes on data loading', url)}."
            )

        self._analyze_distributed_metrics()
        self._analyze_gpu_metrics()

        if self.device_props:
            # Tensor Cores feature is available on GPU cards with compute capability >= 7.0
            # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
            major = self.device_props[0].get('computeMajor')
            # If it's a pure CPU run, then self.tc_used_ratio is None, this rule will not be triggered.
            if (major is not None and major >= 7 and
                    self.tc_used_ratio == 0.0 and
                    self.tc_eligible_ops_kernel_ratio > 0.0):
                url = 'https://pytorch.org/docs/stable/amp.html'
                self.recommendations.append(
                    f'Kernels with {round(self.tc_eligible_ops_kernel_ratio * 100)}%'
                    ' time are launched by Tensor Cores eligible operators. '
                    f"You could enable {href('Automatic Mixed Precision', url)} to speedup by using FP16.")

            # Memory related
            if self.memory_snapshot:
                for (dev_type, dev_id), peak_mem in self.memory_snapshot.get_peak_memory().items():
                    if dev_type == -1:  # ignore cpu
                        continue
                    total_mem = self.device_props[dev_id].get('totalGlobalMem')
                    if total_mem is not None and peak_mem > total_mem * 0.9:
                        percentage = peak_mem / total_mem * 100
                        total_mem_gb = total_mem / 1024 / 1024 / 1024
                        ckp_url = 'https://pytorch.org/docs/stable/checkpoint.html'
                        amp_url = 'https://pytorch.org/docs/stable/amp.html'
                        self.recommendations.append(
                            f'Device memory usage is at the limit of device memory capacity '
                            f'({percentage:.1f}% of {total_mem_gb:.1f}GB on GPU{dev_id}). '
                            'To get better value of your GPU or to use larger batch size for training, please refer to '
                            f"{href('Gradient Checkpoint', ckp_url)} or {href('Automatic Mixed Precision', amp_url)}.")
                        break

    def _analyze_distributed_metrics(self):
        if self.use_dp and len(self.used_devices) > 1:
            url = 'https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead'
            self.recommendations.append(
                f"It is recommended to {href('use DistributedDataParallel instead of DataParallel', url)}"
                ' to do multi-GPU training.')

        if self.use_ddp and CommLibTypes.Nccl not in self.comm_lib and self.device_props:
            for device_prop in self.device_props:
                major = device_prop.get('computeMajor')
                minor = device_prop.get('computeMinor')
                if major is None or minor is None:
                    continue
                compute_capability = '{}.{}'.format(major, minor)
                if float(compute_capability) >= 3.5:
                    text = (
                        'Nccl backend is currently the fastest and highly recommended backend'
                        ' when using DDP for training.')
                    self.recommendations.append(text)
                    break

        communication_ratio = self.avg_costs.costs[ProfileRole.Communication] / self.avg_costs.costs[ProfileRole.Total]
        if communication_ratio > 0.1:
            percentage = communication_ratio * 100
            compress_url = 'https://pytorch.org/docs/stable/ddp_comm_hooks.html',
            grad_acc_url = 'https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa'
            lamb_url = 'https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedLAMB'
            self.recommendations.append(
                f'This run has high time cost on communication. {percentage:.1f}% of the step time is in '
                f"communication. You could try {href('Gradient Compression', compress_url)} or "
                f"{href('Gradient Accumulation', grad_acc_url)} or increase the batch size. "
                'Note: Gradient accumulation will increase global effective batch size, which may hurt model '
                f"convergence and accuracy. For such case, you may want to evaluate {href('LAMB optimizer', lamb_url)}."
            )

    def _memory_events(self) -> List[MemoryEvent]:
        memory_events = [e for e in self.events if e.type == EventTypes.MEMORY]
        memory_events.sort(key=lambda e: e.ts)
        return memory_events
    
    def _clean_memory_events(self) -> List[MemoryEvent]:
        memory_events = [e for e in self.clean_events if e.type == EventTypes.MEMORY]
        memory_events.sort(key=lambda e: e.ts)
        return memory_events

    def _analyze_gpu_metrics(self):
        def get_gpus_str(gpus):
            gpu_list_str = str(gpus[0])
            for i in range(1, len(gpus)):
                if i == len(gpus) - 1:
                    gpu_list_str += 'and {}'.format(gpus[i])
                else:
                    gpu_list_str += ', {}'.format(gpus[i])
            has_str = 'has' if len(gpu_list_str) == 1 else 'have'
            return gpu_list_str, has_str

        low_util_gpus = []
        for gpu_id in self.gpu_metrics_parser.gpu_ids:
            if self.gpu_metrics_parser.gpu_utilization[gpu_id] < 0.5:
                low_util_gpus.append(gpu_id)
        if len(low_util_gpus) > 0:
            gpu_list_str, has_str = get_gpus_str(low_util_gpus)
            text = 'GPU {} {} low utilization. You could try to ' \
                   'increase batch size to improve. Note: Increasing batch size ' \
                   'may affect the speed and stability of model convergence.'.format(gpu_list_str, has_str)
            self.recommendations.append(text)

    def data_clean(self):
        """
        Clean the data in the profile, only keep the data that is after the last profiler step.
        """
        # find the last ProfilerStep start time
        new_profiler_start_ts = 0
        profiler_step = None
        for e in self.events:
            if isinstance(e, trace.ProfilerStepEvent):
                new_profiler_start_ts = max(new_profiler_start_ts, e.ts)
        
        root_node = next(iter(self.tid2tree.values()))
        
        if 'ProfilerStep#' in root_node.children[-1].name:
            self.clean_root_node = root_node.children[-2]
        else:
            self.clean_root_node = root_node.children[-1]
            
        ts = self.clean_root_node.start_time
        te = self.clean_root_node.end_time
        self.clean_events = [e for e in self.events if e.ts >= ts and e.ts <= te]
        
        # self.clean_profiler_start_ts = ts
        
        # self.clean_tid2tree
        
        # memory_events = self._clean_memory_events()
        # if memory_events:
        #     memory_parser = MemoryParser(memory_events)
        #     self.clean_memory_snapshot = memory_parser.find_memory_nodes(self.clean_tid2tree)
        
        # self.forward_backward_events = self._clean_memory_events()


    def data_clean_tree(self):
        """
        Clean the tree, only keep the data that is after the last profiler step.
        """
        if len(self.steps_names) > 1 and self.steps_names[-1] == self.steps_names[-2]:
            self.last_step_name = self.steps_names[-1]
            self.last_step_ts = min(self.steps[-1][0], self.steps[-2][0])
            self.last_step_te = max(self.steps[-1][1], self.steps[-2][1])
        else:
            self.last_step_name = self.steps_names[-1]
            self.last_step_ts = self.steps[-1][0]
            self.last_step_te = self.steps[-1][1]
        
        for tid, root_node in self.tid2tree.items():
            if 'autograd' in root_node.children[0].name:
                # it is backward tree
                new_tree_root = copy.deepcopy(root_node)
                self.clean_tid2tree[tid] = new_tree_root
                new_tree_root.children = []
                
                new_ts = float('inf')
                new_te = 0
                new_duration = 0
                for child in root_node.children:
                    if child.end_time < self.last_step_ts or child.start_time > self.last_step_te:
                        # remove the child
                        continue
                    
                    new_tree_root.children.append(child)
                    new_ts = min(new_ts, child.start_time)
                    new_te = max(new_te, child.end_time)
                    new_duration += child.end_time - child.start_time
                
                new_tree_root.start_time = new_ts
                new_tree_root.end_time = new_te
                print(f"  Clamp duration: {new_te - new_ts}")
                print(f"  New Duration: {new_duration}")
                         
                continue
            
            is_main_thread = False
            for child in root_node.children:
                if 'nn.Module' in child.name:
                    is_main_thread = True
                    break
            
            if is_main_thread:
                # it is main thread
                new_tree_root = copy.deepcopy(root_node)
                self.clean_tid2tree[tid] = new_tree_root
                new_tree_root.children = []
                
                new_ts = float('inf')
                new_te = 0
                new_duration = 0
                for child in root_node.children:
                    if child.end_time < self.last_step_ts or child.start_time > self.last_step_te:
                        # remove the child
                        continue
                    
                    new_tree_root.children.append(child)
                    new_ts = min(new_ts, child.start_time)
                    new_te = max(new_te, child.end_time)
                    new_duration += child.end_time - child.start_time
                
                new_tree_root.start_time = new_ts
                new_tree_root.end_time = new_te
                print(f"  Clamp duration: {new_te - new_ts}")
                print(f"  New Duration: {new_duration}")
                    
                continue
        
        for tid, root_node in self.clean_tid2tree.items():
            node_count, node_classes, node_names, earlist_ts, latest_te = self.statistic_tree(root_node)
            print(f"Thread {tid}:")
            print(f"  Node Count: {node_count}")
            print(f"  Node Classes: {node_classes}")
            # print(f"  Node Names: {node_names}")
            print(f"  Earliest Start Time: {earlist_ts}")
            print(f"  Latest End Time: {latest_te}")
            
    def clean_memory_record(self):
        """
        Clean the memory record, only keep the data that is after the last profiler step.
        """
        self.clean_events = [e for e in self.events if e.ts >= self.last_step_ts and e.ts <= self.last_step_te]
        self.clean_memory_events = self._clean_memory_events()
        memory_parser = MemoryParser(self.clean_memory_events)
        self.clean_memory_snapshot = memory_parser.find_memory_nodes(self.clean_tid2tree)
    
    @staticmethod
    def statistic_tree(root_node):
        node_count = 0
        node_classes = {}
        node_names = {}
        earlist_ts = float('inf')  # Should start with infinity for min comparison
        latest_te = 0  # Should start with 0 for max comparison
        
        def traverse_tree(node):
            nonlocal node_count, node_classes, node_names, earlist_ts, latest_te
            
            if node is None:
                return
            
            node_count += 1
            node_classes[node.__class__.__name__] = node_classes.get(node.__class__.__name__, 0) + 1
            node_names[node.name] = node_names.get(node.name, 0) + 1
            earlist_ts = min(earlist_ts, node.start_time)
            latest_te = max(latest_te, node.end_time)
            
            for child in node.children:
                traverse_tree(child)
            
        traverse_tree(root_node)
        
        return node_count, node_classes, node_names, earlist_ts, latest_te
        