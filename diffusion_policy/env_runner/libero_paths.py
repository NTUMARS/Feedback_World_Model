"""Locate the LIBERO / LIBERO-Plus checkouts used by the simulators.

The two repositories are not Python-packaged dependencies of this code base.
They are expected either on ``PYTHONPATH`` or at the default locations
``<repo_root>/LIBERO`` and ``<repo_root>/LIBERO-plus``, overridable with the
``LIBERO_ROOT`` and ``LIBERO_PLUS_ROOT`` environment variables.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def libero_root() -> str:
    return os.environ.get("LIBERO_ROOT", os.path.join(REPO_ROOT, "LIBERO"))


def libero_plus_root() -> str:
    return os.environ.get("LIBERO_PLUS_ROOT", os.path.join(REPO_ROOT, "LIBERO-plus"))


def add_libero_to_path() -> None:
    """Make the original LIBERO package importable (used by training-time rollouts and data conversion).

    The checkout is put *first* on ``sys.path`` so that it wins over a pip-installed
    LIBERO-Plus (which provides a package of the same name).
    """
    root = libero_root()
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


def add_libero_plus_to_path() -> None:
    """Make the LIBERO-Plus package importable *before* LIBERO (used by ``eval_libero_plus.py``).

    LIBERO and LIBERO-Plus both read ``$LIBERO_CONFIG_PATH/config.yaml``; unless the variable is set,
    LIBERO-Plus is pointed to ``~/.libero_plus`` so that the two configurations do not overwrite each other.
    """
    root = libero_plus_root()
    if not os.path.isdir(root):
        raise FileNotFoundError(f"LIBERO-Plus checkout not found at {root} (set LIBERO_PLUS_ROOT)")
    os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_plus"))
    if root not in sys.path:
        sys.path.insert(0, root)


def get_libero_bddl_root() -> str:
    """Directory containing the ``libero_10/``, ``libero_90/``, ... BDDL files of the original LIBERO."""
    checkout = os.path.join(libero_root(), "libero", "libero", "bddl_files")
    if os.path.isdir(checkout):
        return checkout
    add_libero_to_path()
    from libero.libero import get_libero_path
    return get_libero_path("bddl_files")
