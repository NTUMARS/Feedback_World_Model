"""Latent prediction error of the world model with and without feedback (Fig. 3 / Fig. 11 of the paper).

Reads the per-episode latent logs written by ``eval_robomimic.py`` / ``eval_libero_plus.py``
when ``record_latents=true`` (``<output_dir>/latents/*.npz``).  For every executed action
chunk the log contains

* ``z_obs``            observed latent ``z_t``
* ``z_pred_base``      open-loop prediction ``f_theta(z_t, A_t^(0))``            (Eq. 2)
* ``z_pred_feedback``  feedback-corrected prediction ``z_pred_base + dt * l * e_t`` (Eq. 10)
* ``z_next``           observed next latent ``z_{t+1}``

The one-chunk-ahead prediction error is

    MSE_base(t)     = mean_j (z_pred_base[t, j]     - z_next[t, j])^2
    MSE_feedback(t) = mean_j (z_pred_feedback[t, j] - z_next[t, j])^2

and the relative reduction is ``(MSE_base - MSE_feedback) / MSE_base``.  The script reports
step-pooled means (every executed chunk counts once) with an episode-level bootstrap standard
error, as well as the mean of per-episode means.

Usage::

    python scripts/latent_prediction_mse.py results/robomimic/square/fwm
    python scripts/latent_prediction_mse.py results/robomimic/square/fwm --per-episode
    python scripts/latent_prediction_mse.py results/robomimic/square/fwm --plot 0 --out trajectory.pdf
"""
import argparse
import glob
import os

import numpy as np


def load_episodes(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, 'latents', '*.npz')))
    if not files:
        raise SystemExit(f"no latent logs under {results_dir}/latents (run the evaluation with record_latents=true)")
    episodes = []
    for f in files:
        d = np.load(f)
        episodes.append({
            'name': os.path.splitext(os.path.basename(f))[0],
            'z_obs': d['z_obs'], 'z_next': d['z_next'],
            'z_pred_base': d['z_pred_base'], 'z_pred_feedback': d['z_pred_feedback'],
            'success': bool(d['success']) if 'success' in d else None,
        })
    return episodes


def step_mse(ep):
    base = ((ep['z_pred_base'] - ep['z_next']) ** 2).mean(axis=1)
    feedback = ((ep['z_pred_feedback'] - ep['z_next']) ** 2).mean(axis=1)
    return base, feedback


def bootstrap_se(episodes, n_boot=10000, seed=0):
    """Standard error of the step-pooled mean, resampling whole episodes (clusters)."""
    rng = np.random.default_rng(seed)
    per_ep = [step_mse(ep) for ep in episodes]
    sums = np.array([[b.sum(), f.sum(), len(b)] for b, f in per_ep])
    pooled = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(sums), len(sums))
        s = sums[idx].sum(axis=0)
        pooled.append([s[0] / s[2], s[1] / s[2]])
    pooled = np.array(pooled)
    return pooled.std(axis=0)


def summarize(episodes, per_episode=False):
    all_base, all_fb = [], []
    rows = []
    for ep in episodes:
        b, f = step_mse(ep)
        all_base.append(b)
        all_fb.append(f)
        rows.append((ep['name'], len(b), b.mean(), f.mean(), 100 * (1 - f.mean() / b.mean()), ep['success']))
    all_base, all_fb = np.concatenate(all_base), np.concatenate(all_fb)
    se_base, se_fb = bootstrap_se(episodes)
    reduction = 100 * (1 - all_fb.mean() / all_base.mean())

    print(f"{len(episodes)} episodes, {len(all_base)} executed chunks, latent dim {episodes[0]['z_next'].shape[1]}")
    print(f"step-pooled MSE   base WM     : {all_base.mean():.6f} +/- {se_base:.6f}")
    print(f"step-pooled MSE   feedback WM : {all_fb.mean():.6f} +/- {se_fb:.6f}")
    print(f"relative reduction            : {reduction:.1f}%")
    ep_base = np.array([r[2] for r in rows])
    ep_fb = np.array([r[3] for r in rows])
    print(f"episode-mean MSE  base / feedback : {ep_base.mean():.6f} / {ep_fb.mean():.6f} "
          f"(reduction {100 * (1 - ep_fb.mean() / ep_base.mean()):.1f}%)")
    if per_episode:
        print(f"\n{'episode':<16}{'steps':>6}{'mse_base':>12}{'mse_fb':>12}{'reduction':>11}  success")
        for name, n, b, f, red, succ in rows:
            print(f"{name:<16}{n:>6}{b:>12.6f}{f:>12.6f}{red:>10.1f}%  {succ}")
    return reduction


def pca_projection(reference, n_components=2):
    """Fit a PCA on ``reference`` (T, D); return (projector, explained variance ratio)."""
    mean = reference.mean(axis=0)
    centered = reference - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    ratio = (s ** 2) / (s ** 2).sum()
    basis = vt[:n_components]
    return (lambda z: (z - mean) @ basis.T), ratio[:n_components]


def plot_episode(ep, out_path):
    """3-D view of the latent trajectory: PC1, PC2 and time (Fig. 11 of the paper)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gt = np.concatenate([ep['z_obs'][:1], ep['z_next']], axis=0)           # z_0 ... z_T
    project, ratio = pca_projection(gt)                                     # shared projection
    gt_p = project(gt)
    base_p = project(ep['z_pred_base'])                                     # predictions of z_1 ... z_T
    fb_p = project(ep['z_pred_feedback'])
    t_gt = np.arange(len(gt_p))
    t_pred = np.arange(1, len(gt_p))
    b, f = step_mse(ep)
    reduction = 100 * (1 - f.mean() / b.mean())

    fig = plt.figure(figsize=(5, 4.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(base_p[:, 0], base_p[:, 1], t_pred, color='#1f4e9c', lw=1.6, label='Base WM')
    ax.plot(fb_p[:, 0], fb_p[:, 1], t_pred, color='#d62728', lw=1.6, label='Feedback WM')
    ax.plot(gt_p[:, 0], gt_p[:, 1], t_gt, color='#17becf', lw=1.4, ls='--', label='Ground Truth')
    ax.scatter(gt_p[0, 0], gt_p[0, 1], t_gt[0], marker='*', s=70, color='#17becf', label='Start')
    ax.scatter(gt_p[-1, 0], gt_p[-1, 1], t_gt[-1], marker='^', s=50, color='k', label='End')
    ax.set_xlabel(f'PC1 ({100 * ratio[0]:.1f}% var)')
    ax.set_ylabel(f'PC2 ({100 * ratio[1]:.1f}% var)')
    ax.set_zlabel('Timestep')
    ax.text2D(0.62, 0.92, f'MSE reduction:\n{reduction:.1f}%', transform=ax.transAxes, fontsize=9,
              fontweight='bold', ha='center', bbox=dict(boxstyle='round', fc='white', ec='0.5'))
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=-55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path} (episode {ep['name']}, reduction {reduction:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results_dir', help='evaluation output directory containing latents/*.npz')
    parser.add_argument('--per-episode', action='store_true', help='print one row per episode')
    parser.add_argument('--plot', type=int, default=None, metavar='EPISODE',
                        help='index of the episode (in sorted order) to draw as a 3-D PCA trajectory')
    parser.add_argument('--out', default=None, help='figure path for --plot (default: <results_dir>/latent_trajectory_<i>.pdf)')
    args = parser.parse_args()

    episodes = load_episodes(args.results_dir)
    summarize(episodes, per_episode=args.per_episode)
    if args.plot is not None:
        out = args.out or os.path.join(args.results_dir, f'latent_trajectory_{args.plot}.pdf')
        plot_episode(episodes[args.plot], out)


if __name__ == '__main__':
    main()
