"""
Central seed management for reproducible analysis.

This module provides a unified interface for initializing all random number
generators (RNGs) used across the project, ensuring deterministic results when
the same seed is provided.

Modules seeded:
  - random (Python standard library)
  - numpy.random (NumPy RNG)
  - networkx (graph algorithms)
  - scipy (scientific computing)
  - sklearn (scikit-learn ML)
  - pandas (for reproducible operations like DataFrame.sample())
  - torch (PyTorch, if installed)

Usage:
    from src.seeding import initialize_seeds, get_seed_from_config

    # From config.yaml
    seed = get_seed_from_config(snakemake.config)
    initialize_seeds(seed)

    # Or directly
    initialize_seeds(45)
"""

import random
import numpy as np
from typing import Optional, Dict, Any


def get_seed_from_config(config: Dict[str, Any]) -> int:
    """
    Load the random seed from a config dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically from config.yaml via snakemake)
        Must contain a "seed" key with an integer value.

    Returns
    -------
    int
        The seed value from config["seed"]

    Raises
    ------
    KeyError
        If "seed" key is missing from config
    ValueError
        If seed value is not a valid integer
    """
    if "seed" not in config:
        raise KeyError(
            'Missing "seed" in config. Please add "seed: <integer>" to config.yaml'
        )

    try:
        seed = int(config["seed"])
    except (ValueError, TypeError) as e:
        raise ValueError(
            f'Invalid seed value "{config["seed"]}". Must be an integer.'
        ) from e

    return seed


def initialize_seeds(seed: int) -> None:
    """
    Initialize all random number generators with a given seed.

    This function seeds all major RNG sources in the Python scientific stack:
    - Python's random module
    - NumPy's random module
    - NetworkX (for reproducible graph algorithms)
    - SciPy (for random functions)
    - scikit-learn (for ML algorithms)
    - pandas (for reproducible sampling)
    - PyTorch (if installed)

    Call this function at the start of any script or notebook that uses
    randomness to ensure reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value. Should be non-negative.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If seed is not an integer

    Examples
    --------
    >>> initialize_seeds(42)
    >>> import numpy as np
    >>> np.random.random()  # Will produce consistent value across runs
    """

    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    # Python standard library random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # NetworkX (uses numpy internally, but set explicitly for safety)
    try:
        import networkx as nx
        nx.utils.random_sequence.random.seed(seed)
    except (ImportError, AttributeError):
        pass  # NetworkX not installed or different version

    # SciPy (shares numpy's RNG state)
    try:
        from scipy.special import random as scipy_random
        scipy_random.seed(seed)
    except (ImportError, AttributeError):
        pass

    # scikit-learn (uses numpy internally)
    try:
        from sklearn.utils import random as sklearn_random
        sklearn_random.seed(seed)
    except (ImportError, AttributeError):
        pass

    # pandas (uses numpy for most operations, but explicitly set)
    try:
        import pandas as pd
        # pandas doesn't have a global seed, but np.random.seed affects it
    except ImportError:
        pass

    # PyTorch (if installed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not installed

    # Optional: TensorFlow (if installed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # TensorFlow not installed
