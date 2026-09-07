from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from scipy.signal import find_peaks
import sys
from pathlib import Path

#font = ImageFont.truetype(str(Path.cwd() / "times-new-roman" / "times.ttf"), 200)
#font = ImageFont.truetype(str(Path.cwd() / "arial" / "ARIAL.ttf"), 200)
#font = ImageFont.truetype(str(Path.cwd() / "amatic" / "Amatic-Bold.ttf"), 200)
#font = ImageFont.truetype(str(Path.cwd() / "chunkfive" / "Chunk Five Print.otf"), 80)

bars_per_letter = {
    "A": 2,
    "B": 2,
    "C": 2, # good
    "D": 2,
    "E": 2,
    "F": 1,
    "G": 2, # good
    "H": 2,
    "I": 1,
    "J": 1, # good
    "K": 1,
    "L": 1,
    "M": 3,
    "N": 2,
    "O": 2, # good
    "P": 2,
    "Q": 2, # good
    "R": 2,
    "S": 2, # good
    "T": 1, # good
    "U": 2, # Good
    "V": 1,
    "W": 3, # one more
    "X": 1, # offset
    "Y": 1, # good
    "Z": 2 # not good
}

letters_dir = Path.cwd() / "new_models" / "chunkfive" / "letters"
bars_dir = Path.cwd() / "new_models" / "chunkfive" / "bars"

#good_letters = "CGJOQSTUY" # arial
#bad_letters = "AMSVWX" # amatic
bad_letters = "ABDFHJKLMOPUVWXYZ"

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

    img = Image.new("L", (256,256), 255)
    draw = ImageDraw.Draw(img)

    draw.text((50,10), letter, font=font, fill=0)
    img = np.array(img)
    print('shape before:', img.shape)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # remove white margin space
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)

    cropped_letter = img[y:y+h, x:x+w]
    pad = 5
    # add padding
    cropped_letter = cv2.copyMakeBorder(
        cropped_letter,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=255
    )
    cropped_thresh = thresh[y:y+h, x:x+w] # just the letter but black/white inverse
    cropped_thresh = cv2.copyMakeBorder(
        cropped_thresh,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=255
    )

    letter_bgr = cv2.cvtColor(cropped_letter, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(letters_dir / f"{letter}.png"), letter_bgr)

    bars = np.zeros_like(cropped_thresh)

    #if letter in exception_letter:
    #if letter not in good_letters:
    if letter in bad_letters:

        print('shape:', bars.shape)

        # good letters: CGJOQSTUY
        if letter == "A":
            print("true")
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255
        elif letter == "B":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255            
        elif letter == "D":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255
        elif letter == "F":
            bars[:, 10 : 20] = 255
            bars[:, 40 : 50] = 255
        elif letter == "H":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255            
        elif letter == "J":
            bars[:, 10 : 20] = 255
        elif letter == "K":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255
        elif letter == "L":
            bars[:, 10 : 20] = 255
        elif letter == "M":
            bars[:, 10 : 20] = 255
            bars[:, 35 : 45] = 255
            bars[:, 65 : 75] = 255
        elif letter == "O":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255
        elif letter == "P":
            bars[:, 10 : 20] = 255
            bars[:, 40 : 50] = 255
        elif letter == "U":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255
        elif letter == "V":
            bars[:, 10 : 20] = 255
            bars[:, 50 : 60] = 255
        elif letter == "W":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255
            bars[:, 70 : 80] = 255
        elif letter == "X":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255
        elif letter == "Y":
            bars[:, 30 : 40] = 255
        elif letter == "Z":
            bars[:, 10 : 20] = 255
            bars[:, 45 : 55] = 255

        

        """
        for times new roman:
        if letter == "A":
            bars[:, 110 : 122] = 255
            bars[:, 20 : 32] = 255
        elif letter == "V":
            bars[:, 20 : 32] = 255
            bars[:, 110 : 122] = 255
        elif letter == "W":
            bars[:, 20 : 32] = 255
            bars[:, 100 : 112] = 255
            bars[:, 160 : 172] = 255
        elif letter == "X":
            bars[:, 20 : 32] = 255
            bars[:, 110 : 122] = 255
        elif letter == "Z":
            bars[:, 20 : 32] = 255
            bars[:, 100 : 112] = 255
        """

        """for arial:
        if letter == "A":
            bars[:, 110 : 122] = 255
            bars[:, 20 : 32] = 255
        elif letter == "B":
            bars[:, 100 : 112] = 255
            bars[:, 10 : 22] = 255
        elif letter == "D":
            bars[:, 110 : 122] = 255
            bars[:, 10 : 22] = 255
        elif letter == "E":
            bars[:, 100 : 112] = 255
            bars[:, 10 : 22] = 255
        elif letter == "F":
            bars[:, 90 : 102] = 255
            bars[:, 10 : 22] = 255
        elif letter == "H":
            bars[:, 100 : 112] = 255
            bars[:, 10 : 22] = 255
        elif letter == "I":
            bars[:, 8 : 20] = 255
        elif letter == "K":
            bars[:, 100 : 112] = 255
            bars[:, 10 : 22] = 255
        elif letter == "L":
            bars[:, 8 : 20] = 255
        elif letter == "M":
            bars[:, 65 : 77] = 255
            bars[:, 10 : 22] = 255
            bars[:, 125 : 137] = 255
        elif letter == "N":
            bars[:, 100 : 112] = 255
            bars[:, 10 : 22] = 255
        elif letter == "P":
            bars[:, 8 : 20] = 255
            bars[:, 100 : 112] = 255
        elif letter == "R":
            bars[:, 8 : 20] = 255
            bars[:, 100 : 112] = 255
        elif letter == "V":
            bars[:, 8 : 20] = 255
            bars[:, 115 : 127] = 255
        elif letter == "W":
            bars[:, 90 : 102] = 255
            bars[:, 10 : 22] = 255
            bars[:, 165 : 177] = 255
        elif letter == "X":
            bars[:, 10 : 22] = 255
            bars[:, 110 : 122] = 255
        elif letter == "Z":
            bars[:, 8 : 20] = 255
            bars[:, 100 : 112] = 255
        """

        print(letter_bgr.shape)
        print(bars.shape)
        letter_bgr[bars == 255] = (0, 0, 255)        

    else:

        col_sum = np.sum(cropped_thresh > 0, axis=0) # shape 256
        print('col sum:'); print(col_sum)
        peaks, _ = find_peaks(col_sum, distance=10) # peaks = x-coordinates of most-pixel columns
        print('peaks:', peaks)
        num_bars = bars_per_letter[letter]
        peak_strength = col_sum[peaks]
        top_indices = np.argsort(peak_strength)[-num_bars:]
        selected_peaks = peaks[top_indices]
        print('selected peaks:', selected_peaks)
        
        bar_width = 10
        for p in selected_peaks:
            bars[:, p-bar_width//2 : p+bar_width//2] = 255

    cv2.imwrite(str(bars_dir / f"{letter}.png"), bars)

    letter_bgr[bars == 255] = (0, 0, 255)
    #cv2.imshow("bars overlay", letter_bgr)
    #cv2.waitKey(0)

    cv2.imwrite(str(Path.cwd() / "new_models" / "chunkfive" / "letters_w_bars" / f"{letter}_cap.png"), letter_bgr)

    #cv2.imshow("bars", bars)
    #cv2.waitKey(0)