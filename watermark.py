import cv2
import numpy as np

_WATERMARK_TEXT = "AI GENERATED"


def add_disclosure_watermark(image_bgr: np.ndarray) -> np.ndarray:
    """Return a copy of image_bgr with a visible AI-disclosure watermark."""
    watermarked = image_bgr.copy()
    height, width = watermarked.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(width, height) / 400
    thickness = max(1, int(font_scale * 2))
    (text_width, text_height), _ = cv2.getTextSize(
        _WATERMARK_TEXT, font, font_scale, thickness
    )

    margin = 10
    bar_height = text_height + 2 * margin
    overlay = watermarked.copy()
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, watermarked, 0.4, 0, watermarked)

    text_x = max(margin, (width - text_width) // 2)
    text_y = height - margin - 2
    cv2.putText(
        watermarked,
        _WATERMARK_TEXT,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return watermarked
