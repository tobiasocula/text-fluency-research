import numpy as np
import cv2
from pathlib import Path
from extract_letters import extract_letters
import sys
from match_letter import match_letter

# Load your image
bars = cv2.imread(str(Path.cwd() / "data" / "bars" / "Lbarc1.png"))
img = cv2.imread(str(Path.cwd() / "data" / "text" / "Ltype1.png"))

letters, _ = extract_letters(str(Path.cwd() / "data" / "text" / "Ltype1.png"))

result_img = img.copy()
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

for letterdata in letters:
    letter = letterdata[0]
    x, y, w, h = letterdata[1]

    best_match, matches, scores, corresponding_bar = match_letter(
        letter=letter,
        training_data_dir=Path.cwd() / "data" / "cropped_chars_resized",
        bars_dir=Path.cwd() / "data" / "cropped_bars_resized"
    )

    # Threshold the letter to create a mask (black text = 0, white background = 255)
    _, text_mask = cv2.threshold(letter, 100, 255, cv2.THRESH_BINARY_INV)

    # best_match and corresp_bar is of size (32, 32)
    wanted_shape = letter.shape
    bar_correct_size = cv2.resize(corresponding_bar, wanted_shape)
    #bar_correct_size = cv2.resize(corresponding_bar, (w, h))
    bar_correct_size = bar_correct_size.T  # Transpose to match the target shape

    # Create a colored version of the bar (e.g., red)
    colored_bar = np.zeros_like(bar_correct_size)
    if len(colored_bar.shape) == 2:  # If grayscale
        colored_bar = cv2.cvtColor(colored_bar, cv2.COLOR_GRAY2BGR)
    colored_bar[:] = (0, 0, 255)  # Red in BGR format

    # Create a 3D mask for the colored bar
    mask_3d = np.repeat(text_mask[:, :, np.newaxis], 3, axis=2)

    # Apply the colored bar only to the masked regions
    result_img[y:y+h, x:x+w][mask_3d == 0] = colored_bar[mask_3d == 0]

    #cv2.imshow("match", match)
    #cv2.waitKey(0)

cv2.imshow("result", result_img)
cv2.waitKey(0)

#cv2.imshow("best match", best_match)
#cv2.waitKey(0)