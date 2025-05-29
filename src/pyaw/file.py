"""File system and HDF5 utility functions."""

import logging
from pathlib import Path
from typing import Optional, Union

import h5py

logger = logging.getLogger(__name__)


def get_project_root(marker: str = ".git") -> Path:
    """Locates project root directory by searching for a marker file/folder.

    Args:
        marker: Name of file/directory identifying project root (default: .git)

    Returns:
        Path: Absolute path to project root directory

    Raises:
        FileNotFoundError: If marker not found in directory hierarchy
    """
    current_path = Path(__file__).resolve()
    max_depth = 10  # Prevent infinite loops

    for _ in range(max_depth):
        if (current_path / marker).exists():
            logger.debug("Project root identified at: %s", current_path)
            return current_path
        if current_path.parent == current_path:
            break  # Reached filesystem root
        current_path = current_path.parent

    raise FileNotFoundError(f"Project root marker '{marker}' not found in hierarchy")


def print_hdf5_structure(file_path: Union[str, Path]) -> None:
    """Prints hierarchical structure of an HDF5 file.

    Args:
        file_path: Path to HDF5 file

    Raises:
        FileNotFoundError: If specified file doesn't exist
        ValueError: If file is not valid HDF5 format
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    try:
        with h5py.File(path, 'r') as h5file:
            logger.info("HDF5 structure for: %s", path.name)
            h5file.visititems(lambda name, obj: print(f"{name} ({type(obj).__name__})"))
    except OSError as e:
        raise ValueError(f"Invalid HDF5 file: {path}") from e


def find_parent_directory(
        start_path: Union[str, Path],
        target_dir: str,
        max_depth: int = 6
) -> Optional[Path]:
    """Searches directory hierarchy for target directory.

    Args:
        start_path: Starting directory path
        target_dir: Name of directory to find
        max_depth: Maximum directory levels to search (default: 6)

    Returns:
        Path | None: Absolute path to target directory if found

    Example:
        >>> find_parent_directory(Path.cwd(), 'data')
        PosixPath('/project/data')
    """
    current_path = Path(start_path).resolve()

    for _ in range(max_depth):
        candidate = current_path / target_dir
        if candidate.is_dir():
            return candidate
        if current_path.parent == current_path:
            break  # Reached filesystem root
        current_path = current_path.parent

    return None