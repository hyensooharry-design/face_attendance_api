import os

import numpy as np

os.environ["DUMMY_MODE"] = "1"

from api.embedding import get_embedding_from_image_bytes


def test_dummy_embedding_is_deterministic_and_normalized():
    a1 = get_embedding_from_image_bytes(b"image-a")
    a2 = get_embedding_from_image_bytes(b"image-a")
    b = get_embedding_from_image_bytes(b"image-b")

    assert a1.shape == (512,)
    assert np.allclose(a1, a2)
    assert not np.allclose(a1, b)
    assert np.isclose(np.linalg.norm(a1), 1.0, atol=1e-5)
