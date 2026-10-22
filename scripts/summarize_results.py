"""Collect success rates from evaluation outputs into one table.

Scans a results root for ``eval_results.json`` (Robomimic) and ``summary.json``
(LIBERO-Plus) files and prints a table; optionally writes a CSV.  With
``--fisher A B`` it reports Fisher's exact test between two result directories.

Usage::

    python scripts/summarize_results.py results/
    python scripts/summarize_results.py results/ --csv results/summary.csv
    python scripts/summarize_results.py --fisher results/robomimic/square/base results/robomimic/square/fwm
"""
import argparse
import csv
import json
import os


def _robomimic_row(path):
    with open(path) as f:
        d = json.load(f)
    successes = [v for k, v in d.items() if k.startswith('test/success_')]
    n = len(successes)
    k = int(sum(bool(v) for v in successes))
    rate = k / n if n else float(d.get('test/mean_score', float('nan')))
    return {'run': os.path.dirname(path), 'mode': d.get('mode', ''), 'success': k, 'total': n, 'success_rate': rate}


def _libero_row(path):
    with open(path) as f:
        d = json.load(f)
    o = d['overall']
    return {'run': os.path.dirname(path), 'mode': d.get('mode', ''), 'success': o['success'],
            'total': o['total'], 'success_rate': o['success_rate']}


def collect(root):
    rows = []
    for dirpath, _, files in os.walk(root):
        if 'eval_results.json' in files:
            rows.append(_robomimic_row(os.path.join(dirpath, 'eval_results.json')))
        elif 'summary.json' in files:
            rows.append(_libero_row(os.path.join(dirpath, 'summary.json')))
    return sorted(rows, key=lambda r: r['run'])


def _counts(run_dir):
    for name in ('eval_results.json', 'summary.json'):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            row = _robomimic_row(path) if name == 'eval_results.json' else _libero_row(path)
            return row['success'], row['total']
    raise FileNotFoundError(f"no results in {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', nargs='?', default='results')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--fisher', nargs=2, metavar=('RUN_A', 'RUN_B'))
    args = parser.parse_args()

    if args.fisher:
        from scipy.stats import fisher_exact
        (ka, na), (kb, nb) = _counts(args.fisher[0]), _counts(args.fisher[1])
        _, p = fisher_exact([[ka, na - ka], [kb, nb - kb]])
        print(f"{args.fisher[0]}: {ka}/{na} = {ka / na:.3f}")
        print(f"{args.fisher[1]}: {kb}/{nb} = {kb / nb:.3f}")
        print(f"Fisher's exact test p = {p:.4f}")
        return

    rows = collect(args.root)
    if not rows:
        print(f"no results under {args.root}")
        return
    width = max(len(r['run']) for r in rows)
    print(f"{'run':<{width}}  {'mode':<9} {'success':>8} {'total':>6} {'rate':>7}")
    for r in rows:
        print(f"{r['run']:<{width}}  {r['mode']:<9} {r['success']:>8} {r['total']:>6} {r['success_rate']:>7.3f}")
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == '__main__':
    main()
