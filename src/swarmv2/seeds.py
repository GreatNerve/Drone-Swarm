"""Seed discipline: training, validation (model selection) and the untouched held-out test never overlap.

Training uses every seed outside the reserved ranges, so thousands of parallel worlds do not keep
replaying the same few procedural terrains.
"""
import numpy as np

VALIDATION = range(1000, 2000)
TEST_START = 100_000
TRAIN_RANGES = (range(0, 1000), range(2000, TEST_START))
SPLITS = ("train", "validation", "test")


def split_of(seed):
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"Seeds are nonnegative, got {seed}")
    if seed >= TEST_START:
        return "test"
    return "validation" if seed in VALIDATION else "train"


def check(seed, split):
    """Raise if `seed` does not belong to `split` (catches evaluating on training seeds and similar mistakes)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    actual = split_of(seed)
    if actual != split:
        raise ValueError(f"Seed {seed} is a {actual} seed, not a {split} seed")
    return int(seed)


def training_seeds(rng, count):
    """`count` training seeds drawn uniformly from all training ranges."""
    sizes = np.array([len(r) for r in TRAIN_RANGES])
    offsets = rng.integers(0, sizes.sum(), count)
    first = len(TRAIN_RANGES[0])
    return np.where(offsets < first, offsets, offsets - first + TRAIN_RANGES[1].start).astype(np.int64)
