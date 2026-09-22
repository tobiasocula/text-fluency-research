import cv2
from PIL import Image, ImageDraw, ImageFont
import json

import textwrap

import sys
from pathlib import Path
import numpy as np


def extract_letters(img):
    img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Dilate instead of erode: this bridges the gap between a dot and the
    # letter body beneath it so they form one connected component.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))  # taller than wide
    processed = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 2 and h > 4:   # loosen the height floor so dots aren't excluded pre-merge
            letter = img[y:y+h, x:x+w]  # crop from the ORIGINAL (undilated) image
            letters.append((letter, (x, y, w, h)))

    return letters, img

def non_max_suppression(matches, overlap_thresh=0.3):
    """matches: list of (x, y, w, h, score). Returns filtered list."""
    if not matches:
        return []
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h, s in matches], dtype=float)
    scores = np.array([s for *_, s in matches])
    order = scores.argsort()[::-1]

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= overlap_thresh]

    return [matches[i] for i in keep]

def extract_letters_template(img, text, letters_dir,
                              template_font_size, target_font_size,
                              match_thresh=0.7):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    unique_chars = set(text) - {" ", "\n"}

    by_char = {}
    for ch in unique_chars:
        template = load_scaled_template(ch, letters_dir, template_font_size, target_font_size)
        if template is None:
            continue

        th, tw = template.shape
        if th == 0 or tw == 0 or th > gray.shape[0] or tw > gray.shape[1]:
            continue

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= match_thresh)
        matches = [(x, y, tw, th, result[y, x]) for x, y in zip(xs, ys)]
        by_char[ch] = non_max_suppression(matches, overlap_thresh=0.3)

    letters = [] # PER CHARACTER: STORE LETTER MATCHES AND SCORE
    for ch, matches in by_char.items():
        for x, y, w, h, score in matches:
            letter_img = gray[y:y+h, x:x+w]
            letters.append((letter_img, (x, y, w, h), ch, score))

    letters.sort(key=lambda l: l[1][0])  # left-to-right; bucket by row if multi-line
    letters = group_by_position(letters)

    return letters, gray

def group_by_position(letters, x_gap_thresh=None):
    """
    letters: list of (crop, (x, y, w, h), ch, score), already sorted left-to-right
    Groups overlapping/adjacent detections into clusters (one cluster per
    actual letter position), then keeps only the highest-scoring detection
    per cluster.
    """
    if not letters:
        return []

    groups = []
    current_group = [letters[0]]

    for item in letters[1:]:
        _, (x, y, w, h), _, _ = item
        _, (px, py, pw, ph), _, _ = current_group[-1]

        # does this box overlap (or sit very close to) the previous one on the x-axis?
        prev_right = px + pw
        gap = x - prev_right
        threshold = x_gap_thresh if x_gap_thresh is not None else 0.3 * pw

        if gap <= threshold:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]

    groups.append(current_group)

    # per group, keep only the highest-scoring detection
    best_per_group = []
    for group in groups:
        best = max(group, key=lambda l: l[3])  # l[3] is score
        best_per_group.append(best)

    return best_per_group

def remove_diacritics(text):
    mapping = {
        "ä": "a",
        "à": "a",
        "á": "a",
        "å": "a",
        "è": "e",
        "é": "e",
        "ë": "e",
        "ï": "i",
        "í": "i",
        "ű": "u",
        "ü": "u",
        "ú": "u",
        "ó": "o",
        "ö": "o",
        "ñ": "n"
    }
    def convert(char):
        x = mapping.get(char, "")
        if x:
            return x
        return char

    return "".join([convert(x) for x in text])



from PIL import Image, ImageDraw, ImageFont

def render_char_template(ch, font, pad=2):
    """Render a single character at the given font, tightly cropped, grayscale."""
    bbox = font.getbbox(ch)
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("L", (max(w, 1), max(h, 1)), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), ch, font=font, fill=0)
    return np.array(img)


def match_letter(letter, font, bars_dir, candidate_chars):
    """
    letter: grayscale crop of a single detected letter (from extract_letters)
    font: the same PIL ImageFont used to render the source image
    bars_dir: dir of bar PNGs, one per character
    candidate_chars: iterable of characters to test against, e.g. the
                      alphabet plus any diacritic variants you care about
    """
    lh, lw = letter.shape[:2]

    scores = {}
    for ch in candidate_chars:
        template = render_char_template(ch, font)
        th, tw = template.shape

        # Align on a shared canvas rather than resizing either one —
        # both are already at the correct, matching scale, so we just
        # need a common frame for matchTemplate to compare within.
        canvas_h, canvas_w = max(th, lh) + 4, max(tw, lw) + 4
        letter_canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
        letter_canvas[0:lh, 0:lw] = letter
        templ_canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
        templ_canvas[0:th, 0:tw] = template

        result = cv2.matchTemplate(letter_canvas, templ_canvas, cv2.TM_CCOEFF_NORMED)
        scores[ch] = float(result.max())

    best_match = max(scores, key=scores.get)

    bar_filename = f"{best_match}.png" if not best_match.isupper() else f"{best_match}_cap.png"
    corresponding_bar = cv2.imread(str(bars_dir / bar_filename), cv2.IMREAD_UNCHANGED)

    return best_match, list(scores.keys()), scores, corresponding_bar


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

def load_scaled_template(ch, letters_dir, template_font_size, target_font_size):
    letter_path = f"{ch}.png" if not ch.isupper() else f"{ch}_cap.png"
    template = cv2.imread(str(letters_dir / letter_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        return None

    scale = target_font_size / template_font_size
    th, tw = template.shape
    new_w = max(1, round(tw * scale))
    new_h = max(1, round(th * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(template, (new_w, new_h), interpolation=interp)


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