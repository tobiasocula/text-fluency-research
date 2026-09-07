import numpy as np
import cv2
from pathlib import Path
from extract_letters import extract_letters
from match_letter import match_letter

bars = cv2.imread(str(Path.cwd() / "data" / "bars" / "Lbarc4.png"))
img = cv2.imread(str(Path.cwd() / "data" / "text" / "Ltype4.png"))

letters, _ = extract_letters(str(Path.cwd() / "data" / "text" / "Ltype4.png"))

# Ensure result_img is in BGR format
result_img = img.copy()

for letterdata in letters:
    letter = letterdata[0]
    x, y, w, h = letterdata[1]

    best_match, matches, scores, corresponding_bar = match_letter(
        letter=letter,
        training_data_dir=Path.cwd() / "data" / "cropped_chars_resized",
        bars_dir=Path.cwd() / "data" / "cropped_bars_resized"
    )

    # Threshold the letter to create a mask (black text = 0, white background = 255)
    _, text_mask = cv2.threshold(letter, 150, 255, cv2.THRESH_BINARY)

    # Resize the bar to match the letter
    bar_correct_size = cv2.resize(corresponding_bar, (w, h))

    # Convert to grayscale
    if len(bar_correct_size.shape) == 3:
        bar_gray = cv2.cvtColor(bar_correct_size, cv2.COLOR_BGR2GRAY)
    else:
        bar_gray = bar_correct_size

    # Detect black bars (invert threshold)
    _, bar_mask = cv2.threshold(bar_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Combine bar mask with the text mask
    combined_mask = cv2.bitwise_and(bar_mask, text_mask)

    # Paint only those pixels red
    letter_region = result_img[y:y+h, x:x+w].copy()
    letter_region[combined_mask == 255] = (0, 0, 255)

    result_img[y:y+h, x:x+w] = letter_region

cv2.imshow("result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
