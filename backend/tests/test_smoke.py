"""Smoke test for the numeric stack until module-level unit tests land."""

import numpy as np


def test_numpy_vectorization_sanity() -> None:
    v = np.array([3.0, 4.0])
    assert np.isclose(np.linalg.norm(v), 5.0)