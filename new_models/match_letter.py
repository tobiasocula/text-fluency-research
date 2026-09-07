import cv2
from pathlib import Path
import sys
import numpy as np

def match_letter(letter, training_data_dir, bars_dir):
    # Load training data
    training_letters = []
    bars = []

    training_paths = sorted(Path(training_data_dir).glob("*.png"))
    bars_paths = sorted(Path(bars_dir).glob("*.png"))

    for barpath, path in zip(bars_paths, training_paths):
        training_letter = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        bar = cv2.imread(str(barpath), cv2.IMREAD_GRAYSCALE)
        #cv2.imshow("trainign letter", training_letter)
        #cv2.waitKey(0)
        training_letters.append(training_letter)
        bars.append(bar)

    # Resize the extracted letter to match training data size
    """
    target = 256

    h, w = letter.shape

    scale = target / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(letter, (new_w, new_h))

    canvas = np.ones((target, target), dtype=np.uint8) * 255

    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2

    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    letter_resized = canvas
    """

    #cv2.imshow("letter resized", letter_resized)
    #cv2.waitKey(0)

    # Compare the letter to each training letter using template matching
    best_match = None
    best_score = -1

    corresponding_bar = None

    matches = []
    scores = []

    # print("actual letter")
    # cv2.imshow("actual letter", letter)
    # cv2.waitKey(0)
    # print("letter resized")
    # cv2.imshow("letter_resized", letter_resized)
    # cv2.waitKey(0)

    h, w = letter.shape

    # cv2.imshow("letter", letter)
    # cv2.waitKey(0)

    for target_bar, training_letter in zip(bars, training_letters):
        training_letter_resized = cv2.resize(training_letter, (w, h))
        target_bar_resized = cv2.resize(target_bar, (w, h))

        # cv2.imshow("trainin gletter resized", training_letter_resized)
        # cv2.waitKey(0)
        # cv2.imshow("target bar resized", target_bar_resized)
        # cv2.waitKey(0)

        #score = cv2.matchTemplate(letter_resized, training_letter, cv2.TM_SQDIFF_NORMED)
        #score = -score
        _, letter_bin = cv2.threshold(letter, 200, 255, cv2.THRESH_BINARY_INV)
        _, train_bin  = cv2.threshold(training_letter_resized, 200, 255, cv2.THRESH_BINARY_INV)
        print('FINDING TEMPLATE')

        intersection = np.sum((letter_bin > 0) & (train_bin > 0))
        union = np.sum((letter_bin > 0) | (train_bin > 0))

        score = intersection / union

        # #_, score, _, _ = cv2.minMaxLoc(score)

        # # print('training letter')
        # cv2.imshow("training letter", training_letter)
        # cv2.waitKey(0)
        # print('score:', score)

        scores.append(score)

        if score > best_score:
            best_score = score
            best_match = training_letter
            corresponding_bar = target_bar

        matches.append(training_letter)

    # cv2.imshow("best match", best_match)
    # cv2.waitKey(0)
    # print('best score:', best_score)

    return best_match, matches, scores, corresponding_bar