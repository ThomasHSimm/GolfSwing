"""Utility functions."""
from pathlib import Path


def get_package_path() -> Path:
    """Return the path to the package.

    Parameters:

    Returns:
        Path: path to package
    """
    return Path(__file__).parent.parent
