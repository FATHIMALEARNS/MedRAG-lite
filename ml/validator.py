import cv2
import numpy as np

def is_valid_xray(image_path):
    """
    Validates that an uploaded image is a grayscale medical scan (X-ray).
    Rejects: portraits, colored photos, dark non-medical images.
    """
    img = cv2.imread(image_path)

    if img is None:
        return False

    h, w = img.shape[:2]

    # --- Color Variance Check (only on non-dark pixels) ---
    # Real X-rays are pure grayscale: R ≈ G ≈ B for every pixel.
    # We ignore very dark background pixels (< 40) to prevent night photos
    # with black backgrounds from faking a low variance score.
    brightness = np.mean(img, axis=2)
    bright_mask = brightness > 40
    bright_pixels = img[bright_mask]

    # Image is almost entirely black — not a valid scan
    if len(bright_pixels) < (h * w * 0.05):
        return False

    # Check R/G/B channel divergence on the visible pixels only
    channel_std_per_pixel = np.std(bright_pixels.astype(np.float32), axis=1)
    mean_std = np.mean(channel_std_per_pixel)

    # Threshold 18.0: rejects colorful portraits/photos, accepts blue-tinted
    # digital X-rays and standard grayscale medical scans
    if mean_std > 18.0:
        return False

    return True