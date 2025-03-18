from collections import defaultdict
from typing import List, Dict
from .utils import Canonicalizer, lttb_sample
from .profiler.memory_parser import MemoryRecord


def get_memory_curve(
        time_metric: str = 'ms',
        memory_metric: str = 'K',
        patch_for_step_plot=True,
        profiler_start_ts: float = 0.0,
        memory_records: List[MemoryRecord] = None,
        gpu_infos: Dict[int, Dict[str, float]] = None):
    
    def get_curves_and_peaks(records: List[MemoryRecord], cano: Canonicalizer):
        """Inputs:
            records: Sorted list of MemoryRecord

        For example:
        ```py
        {
            'CPU': [# Timestamp, Total Allocated, Total Reserved, Device Total Memory, operator
                [1, 4, 4, 1000000, 'aten::add'],
                [2, 16, 16, 1000000, "aten::empty],
                [4, 4, 16, 1000000, '...'],
            ],
            'GPU0': ...
        }
        ```"""
        curves = defaultdict(list)
        peaks = defaultdict(float)
        for r in records:
            if r.addr is None:
                continue
            dev = r.device_name
            ts = r.ts
            ta = r.total_allocated
            tr = r.total_reserved

            if ta != ta or tr != tr:  # isnan
                continue

            curves[dev].append([
                cano.convert_time(ts - profiler_start_ts),
                cano.convert_memory(ta),
                cano.convert_memory(tr),
            ])
            peaks[dev] = max(peaks[dev], ta)

        for dev in curves:
            if len(curves[dev]) == 0:
                del curves[dev]
                del peaks[dev]

        return curves, peaks

    # NOTE: this should have been occured in frontend
    def patch_curves_for_step_plot(curves: Dict[str, List]):
        # For example, if a curve is [(0, 0), (1, 1), (2,2)], the line plot
        # is a stright line. Interpolating it as [(0, 0), (1, 0), (1, 1),
        # (2,1) (2,2)], then the line plot will work as step plot.
        new_curves = defaultdict(list)
        for dev, curve in curves.items():
            new_curve = []
            for i, p in enumerate(curve):
                if i != 0:
                    new_curve.append(p[:1] + new_curve[-1][1:])
                new_curve.append(p)
            new_curves[dev] = new_curve
        return new_curves

    cano = Canonicalizer(time_metric, memory_metric)

    curves, peaks = get_curves_and_peaks(memory_records, cano)
    if patch_for_step_plot:
        curves = patch_curves_for_step_plot(curves)
    peaks_formatted = {}
    totals = {}
    for dev, value in peaks.items():
        peaks_formatted[dev] = 'Peak Memory Usage: {:.1f}{}'.format(cano.convert_memory(value), cano.memory_metric)
        if dev != 'CPU':
            try:
                totals[dev] = cano.convert_memory(gpu_infos[int(dev[3:])]['Memory Raw'])
            except BaseException:
                pass

    devices: List[str] = sorted(list(curves.keys()))
    default_device = 'CPU'
    for dev in devices:
        if dev.startswith('GPU'):
            default_device = dev
            break

    curves = lttb_sample(curves)

    return {
        'metadata': {
            'default_device': default_device,
            'devices': devices,
            'peaks': peaks_formatted,
            'totals': totals,
            'first_ts': profiler_start_ts,
            'time_metric': cano.time_metric,
            'memory_metric': cano.memory_metric,
            'time_factor': cano.time_factor,
            'memory_factor': cano.memory_factor,
        },
        'columns': [
            {'name': f'Time ({cano.time_metric})', 'type': 'number', 'tooltip': 'Time since profiler starts.'},
            {'name': f'Allocated ({cano.memory_metric})', 'type': 'number', 'tooltip': 'Total memory in use.'},
            {'name': f'Reserved ({cano.memory_metric})', 'type': 'number',
                'tooltip': 'Total reserved memory by allocator, both used and unused.'},
        ],
        'rows': curves,
    }
