import numpy as np

from watermark import add_disclosure_watermark


def test_add_disclosure_watermark_changes_pixels_without_resizing():
    image = np.zeros((400, 400, 3), dtype="uint8")
    watermarked = add_disclosure_watermark(image)
    assert watermarked.shape == image.shape
    assert watermarked.dtype == image.dtype
    assert not np.array_equal(watermarked, image)
