import cv2
import sys


def extract_letters(image):
    

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    print('EXTRACTING LETTER')
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("thresh", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    processed = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # crop and return individual letters
    letters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 3 and h > 10:  # Filter out noise
            letter = gray[y:y+h, x:x+w]
            letters.append((letter, (x, y, w, h))) # letter and bounding box

    return letters, image
