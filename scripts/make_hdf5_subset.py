"""Extract a subset of demonstrations from a Robomimic/LIBERO HDF5 file into a new file.

Used to create the small validation split for world-model training (the paper
holds out the last 10 demonstrations of each Robomimic task).  Demonstrations are
renumbered ``demo_0, demo_1, ...`` in the output.

Usage::

    python scripts/make_hdf5_subset.py data/square/data/expert_demonstration/square_ph_demo.hdf5 \
        data/square/data/expert_demonstration/square_ph_demo_val.hdf5 --last 10
"""
import argparse

import h5py


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input')
    parser.add_argument('output')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--last', type=int, help='keep the last N demonstrations')
    group.add_argument('--first', type=int, help='keep the first N demonstrations')
    group.add_argument('--indices', type=int, nargs='+', help='explicit demonstration indices')
    args = parser.parse_args()

    with h5py.File(args.input, 'r') as fin, h5py.File(args.output, 'w') as fout:
        keys = sorted([k for k in fin['data'].keys() if k.startswith('demo_')], key=lambda k: int(k[5:]))
        if args.last is not None:
            selected = keys[-args.last:]
        elif args.first is not None:
            selected = keys[:args.first]
        else:
            selected = [f'demo_{i}' for i in args.indices]
        data_out = fout.create_group('data')
        for attr, value in fin['data'].attrs.items():
            data_out.attrs[attr] = value
        total = 0
        for new_idx, key in enumerate(selected):
            fin.copy(f'data/{key}', data_out, name=f'demo_{new_idx}')
            total += int(fin[f'data/{key}'].attrs.get('num_samples', fin[f'data/{key}/actions'].shape[0]))
        data_out.attrs['total'] = total
        for key in fin.keys():
            if key != 'data':
                fin.copy(key, fout)
    print(f"wrote {len(selected)} demonstrations ({total} samples) to {args.output}")


if __name__ == '__main__':
    main()
