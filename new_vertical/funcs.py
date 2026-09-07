import cv2
from PIL import Image, ImageDraw, ImageFont
import json

import textwrap

import sys
from pathlib import Path
import numpy as np


def extract_letters(img):
    

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    processed = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # crop and return individual letters
    letters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 3 and h > 10:  # Filter out noise
            letter = img[y:y+h, x:x+w]
            letters.append((letter, (x, y, w, h))) # letter and bounding box

    return letters, img



def match_letter(letter, training_data_dir, bars_dir):
    # Load training data
    training_letters = []
    bars = []

    training_paths = sorted(Path(training_data_dir).glob("*.png"))
    bars_paths = sorted(Path(bars_dir).glob("*.png"))

    for barpath, path in zip(bars_paths, training_paths):
        training_letter = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        bar = cv2.imread(str(barpath), cv2.IMREAD_GRAYSCALE)
        training_letters.append(training_letter)
        bars.append(bar)

 
    # Compare the letter to each training letter using template matching
    best_match = None
    best_score = -1

    corresponding_bar = None

    matches = []
    scores = []
    h, w = letter.shape


    for target_bar, training_letter in zip(bars, training_letters):
        training_letter_resized = cv2.resize(training_letter, (w, h))
        _, letter_bin = cv2.threshold(letter, 200, 255, cv2.THRESH_BINARY_INV)
        _, train_bin  = cv2.threshold(training_letter_resized, 200, 255, cv2.THRESH_BINARY_INV)
        print('FINDING TEMPLATE')

        intersection = np.sum((letter_bin > 0) & (train_bin > 0))
        union = np.sum((letter_bin > 0) | (train_bin > 0))

        score = intersection / union

        scores.append(score)

        if score > best_score:
            best_score = score
            best_match = training_letter
            corresponding_bar = target_bar

        matches.append(training_letter)

    return best_match, matches, scores, corresponding_bar


def get_valid_texts(jsonl_files, num_texts, max_chars=None):
    lan_texts = []
    for input_file in jsonl_files:
        texts = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Parse JSON and extract text field
                try:
                    record = json.loads(line)
                    text = record.get('text', '')
                    if max_chars is not None:
                        if len(text) > max_chars:
                            text = text[:max_chars]
                    texts.append(text)
                    if len(texts) >= num_texts:
                        break
                    
                    if line_num % 10000 == 0:
                        print(f"Processed {line_num} lines...")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON on line {line_num}")
        lan_texts.append(texts)
    return lan_texts


def text_to_image(
    text,
    font_path=None,
    font_size=48,
    padding=20,
    line_spacing=10,
):
    """
    Render a string to an OpenCV image.

    Returns:
        image (np.ndarray): BGR uint8 image suitable for OpenCV.
    """

    # Load font
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)

    # Measure text
    dummy = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(dummy)

    bbox = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=line_spacing,
    )

    width = bbox[2] - bbox[0] + 2 * padding
    height = bbox[3] - bbox[1] + 2 * padding

    # Create white image
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    # Draw black text
    draw.multiline_text(
        (padding, padding),
        text,
        fill=0,
        font=font,
        spacing=line_spacing,
    )

    # Convert to OpenCV BGR
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

def eval_img(img):
    a = np.array(img)

    height_total, width_total, k = img.shape
    assert k==3, AssertionError()

    mask = (a == [0, 0, 255]).all(axis=2).astype(np.uint8) * 255  # Convert to uint8 and scale to 0-255
    inverted_mask = cv2.bitwise_not(mask)
    inverted_mask = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)

    white_buffer = 2 # both directions (up and down)
    labels = np.full(height_total, np.nan)
    class_counter = 0
    in_white = True
    # None = white, int = class label
    buffer = 2
    for y in range(white_buffer, height_total - white_buffer):
        if np.all(inverted_mask[y-buffer:y+buffer,:] == 255):
            in_white = True
        elif np.any(inverted_mask[y,:] == 0):
            # there is black
            if in_white: # prev was white
                class_counter += 1
            labels[y] = class_counter
            in_white = False

    def differences(contours):
        if len(contours) == 0:
            return []
        
        # Get leftmost and rightmost x for each contour
        leftmost_xs = []
        rightmost_xs = []
        for contour in contours:
            xs = contour[:, 0, 0]
            leftmost_x = np.min(xs)
            rightmost_x = np.max(xs)
            leftmost_xs.append(leftmost_x)
            rightmost_xs.append(rightmost_x)
        
        # Sort by leftmost x (keep paired)
        paired = [(leftmost_xs[i], rightmost_xs[i]) for i in range(len(contours))]
        paired.sort(key=lambda p: p[0])
        
        leftmost_xs = [p[0] for p in paired]
        rightmost_xs = [p[1] for p in paired]
        
        # Calculate gaps between consecutive contours
        d = []
        for i in range(len(leftmost_xs)-1):
            diff = leftmost_xs[i+1] - rightmost_xs[i]
            d.append(diff)

        return d
    
    for cc in range(1,1+class_counter):
        valid_indices = np.where(labels == cc)[0]
        section = inverted_mask[valid_indices,:]
        section = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ds = differences(contours)

    return ds





def match_letter(letter, training_data_dir, bars_dir):

    # Load training data
    training_letters = []
    bars = []

    training_paths = sorted(Path(training_data_dir).glob("*.png"))
    bars_paths = sorted(Path(bars_dir).glob("*.png"))

    for barpath, path in zip(bars_paths, training_paths):
        training_letter = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        bar = cv2.imread(str(barpath), cv2.IMREAD_GRAYSCALE)
        training_letters.append(training_letter)
        bars.append(bar)

    # Compare the letter to each training letter using template matching
    best_match = None
    best_score = -1

    corresponding_bar = None

    matches = []
    scores = []
    h, w = letter.shape
    for target_bar, training_letter in zip(bars, training_letters):
        training_letter_resized = cv2.resize(training_letter, (w, h))
        _, letter_bin = cv2.threshold(letter, 200, 255, cv2.THRESH_BINARY_INV)
        _, train_bin  = cv2.threshold(training_letter_resized, 200, 255, cv2.THRESH_BINARY_INV)

        intersection = np.sum((letter_bin > 0) & (train_bin > 0))
        union = np.sum((letter_bin > 0) | (train_bin > 0))

        score = intersection / union

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