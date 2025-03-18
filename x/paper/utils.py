from torch_tb_profiler import io
import gzip
import io as sysio
import json
import re
import tempfile
from json.decoder import JSONDecodeError
from typing import Dict, List, Optional

def load_json(path):
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
                print('Get JSONDecodeError: %s, Re-encode it to temp file' % e.msg)
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
        fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', dir='./', delete=False)
        fp.close()
        with gzip.open(fp.name, mode='wt') as fzip:
            fzip.write(json.dumps(trace_json))
        trace_path = fp.name

    return trace_path, trace_json
