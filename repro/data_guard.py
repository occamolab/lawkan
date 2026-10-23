"""Small preflight helpers for optional datasets."""
import os


def skip_if_missing(path, label='required data'):
    """Print a SKIP-DATA line and return True when *path* is absent."""
    if os.path.exists(path):
        return False
    print(f'SKIP-DATA: {label} not found at {path}', flush=True)
    return True
