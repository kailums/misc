import json
import argparse
import statistics
import pickle
import statistics
import pandas as pd


def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--input', type=str, help='input a file that contains profile result file names')
    parser.add_argument('--output', type=str, help='output file name')
    parser.add_argument('--desc', action='store_true', default=False)
    parser.add_argument('--sum', action='store_true', default=False)

    args = parser.parse_args()
    return args

def gpu_stat_by_parent(args, gpu_frame):
    # do statistic
    op = gpu_frame.groupby(gpu_frame['op'])
    if args.desc:
        print(op['dur'].describe())
    if args.sum:
        print(op['dur'].sum())

def get_all_gpu_events(events):
    res = []
    for e in events:
        cat = e.get('cat')
        if cat is None:
            continue
        if cat != 'Kernel':
            continue
        args = e.get('args')
        if args is None:
            continue

        dur = e.get('dur')
        name = e.get('name')
        parent_name = args.get('parent_name')
        op_name = args.get('op_name')
        res.append({
            'name': name,
            'node': parent_name,
            'op': op_name,
            'dur': int(dur)
            })

    frame = pd.DataFrame(res)

    return frame

def main(args):
    # get events from json trace file 
    with open(args.input) as fp:
        trace = json.load(fp)
        if isinstance(trace, dict):
            events = trace['traceEvents']
        else:
            assert isinstance(trace, list)
            events = trace

    gpu_events = get_all_gpu_events(events)
    #gpu_events.to_csv(args.output)
    gpu_stat_by_parent(args, gpu_events) 


if __name__ == '__main__':
    args = get_arges()
    main(args)
