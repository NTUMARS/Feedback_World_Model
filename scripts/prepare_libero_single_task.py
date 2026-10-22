"""Create per-task data directories for the LIBERO-10 single-task experiments.

The policy data loader accepts a directory of ``*_demo.hdf5`` files and the
training-time rollout runner globs the same directory, so each single-task
experiment gets its own directory containing exactly one task (a symlink to the
absolute-action demonstration file produced by ``convert_libero_to_abs_actions.py``).

Usage::

    python scripts/prepare_libero_single_task.py --source data/libero_10_abs/libero_10 \
        --tasks KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
                KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it \
                KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \
                LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
"""
import argparse
import os

PAPER_TASKS = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='directory with the absolute-action LIBERO-10 *_demo.hdf5 files')
    parser.add_argument('--destination', default='data/libero_10_single_task')
    parser.add_argument('--tasks', nargs='*', default=PAPER_TASKS)
    parser.add_argument('--copy', action='store_true', help='copy files instead of symlinking')
    args = parser.parse_args()

    for task in args.tasks:
        src = os.path.abspath(os.path.join(args.source, f'{task}_demo.hdf5'))
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        task_dir = os.path.join(args.destination, task)
        os.makedirs(task_dir, exist_ok=True)
        dst = os.path.join(task_dir, f'{task}_demo.hdf5')
        if os.path.lexists(dst):
            os.remove(dst)
        if args.copy:
            import shutil
            shutil.copy(src, dst)
        else:
            os.symlink(src, dst)
        print(f"[ok] {task_dir}")


if __name__ == '__main__':
    main()
