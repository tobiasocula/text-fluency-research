import cv2
import numpy as np



def extract_vertical_red_bars(input_path, output_path):
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read {input_path}")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red wraps around the hue axis, so use two ranges
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 100, 100])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1)
    mask |= cv2.inRange(hsv, lower2, upper2)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Output image
    result = np.zeros_like(img)

    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]

        # Keep only tall, narrow components
        if h > 20 and h > 3 * w:
            result[labels == i] = img[labels == i]

    cv2.imwrite(output_path, result)

from pathlib import Path
bars = Path.cwd() / "new_vertical" / "letters_w_bars"
for b in bars.iterdir():
    extract_vertical_red_bars(b, Path.cwd() / 'new_vertical' / 'bars' / f"{b.name}.png")