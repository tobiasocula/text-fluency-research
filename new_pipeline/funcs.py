import numpy as np
import cv2
from pathlib import Path
import sys
from sklearn.cluster import DBSCAN
import json

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


import zstandard as zstd

def get_valid_texts_zstd(jsonl_files, num_texts):
    lan_texts = []
    for input_file in jsonl_files:
        texts = []
        with open(input_file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                for line_num, raw_line in enumerate(reader):
                    line = raw_line.decode('utf-8').strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get('text', '')
                        texts.append(text)
                        if len(texts) >= num_texts:
                            break
                        if line_num % 10000 == 0:
                            print(f"Processed {line_num} lines from {input_file.name}...")
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON on line {line_num} in {input_file.name}")
        lan_texts.append(texts)
    return lan_texts


def k_means(data, k):
    # data: np-array of vectors (points), shape (n_vectors, vector_dim)
    # k: amount of clusters

    # data size: (n_points, vector_dim)

    center_indices = np.random.randint(0, len(data), size=(1,k)) # (1, k)
    centers = np.array([data[i] for i in center_indices]).reshape((k, data.shape[1])) # (k, vector_dim)
    
    clusters = [[] for _ in range(k)] # (k, n_vectors_to_cluster, vector_dim)
    
    while True:
        new_clusters = clusters
        """
        for every point: determine dist to every cluster
        categorize points based on dist (take smallest)
        redetermine clusters
        """

        labels = [] # length n_vectors

        # determine clusters
        for vector in data:
            # vector: (1, vector_dim)

            # distances for vector to every cluster vector
            # (1, k)

            distances = np.array([np.linalg.norm(vector - core) for core in centers])
            idx = np.argmin(distances)
            new_clusters[idx].append(vector) # list of length k
            labels.append(int(idx))

        if new_clusters == clusters:
            return clusters, centers, labels

        # determine new centers
        for k_idx, cluster in enumerate(clusters):
            # cluster: list of vectors
            avg_vector = np.zeros(data.shape[1])
            for vec in cluster:
                # vec: (vector_dim)
                avg_vector += vec
            avg_vector /= len(cluster)
            centers[k_idx,:] = avg_vector

def extract_letters(image):
    
    gray = image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    print('EXTRACTING LETTER')
    

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    #processed = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # crop and return individual letters
    letters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 3 and h > 10:  # Filter out noise
            letter = gray[y:y+h, x:x+w]
            letters.append((letter, (x, y, w, h))) # letter and bounding box

    return letters

def draw_bar(letter, pixel_weight, edge_weight, width):
    """
    weights: pixel density, symmetry, edge/bend
    width = width of bar
    """
    
    _, thresh = cv2.threshold(letter, 200, 255, cv2.THRESH_BINARY_INV)
    
    def pixel_scores(img):
        h, w = img.shape
        counts = []
        
        for i in range(width, w - width * 2):
            section = img[:, i:i+width]
            count = np.sum(section > 0)
            counts.append(count)
        
        scores_norm = np.array(counts) / (h * w)
        return scores_norm
    
    def edge_scores(img):
        def slope(contours):
            ys, xs = [], []
            for contour in contours:
                for p in contour[:, 0, :]:
                    x, y = p
                    ys.append(y)
                    xs.append(x)
            y_diff = np.diff(ys)
            x_diff = np.diff(xs)
            slopes = y_diff / (x_diff + 1e-7)
            avg_slope = np.mean(slopes)
            return avg_slope
        
        h, w = img.shape
        scores = []
        
        for i in range(width, w - width * 2):
            section = img[:, i:i+width]
            section_left = img[:, i-width:i]
            section_right = img[:, i+width:i+2*width]
            
            contours, _ = cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_left, _ = cv2.findContours(section_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_right, _ = cv2.findContours(section_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            avg_slope = slope(contours)
            avg_slope_left = slope(contours_left)
            avg_slope_right = slope(contours_right)
            
            error = (avg_slope - avg_slope_left)**2 + (avg_slope - avg_slope_right)**2
            scores.append(error)
        
        scores = np.array(scores)

        return scores / np.sum(scores)
    
    def compute_positions(img):
        h, w = img.shape
        xs = []
        for i in range(width, w - width * 2):
            xs.append(i)
        return xs
    
    x_positions = compute_positions(thresh)
    scores_pixels = pixel_scores(thresh)
    scores_edges = edge_scores(thresh)
    
    scores_final = scores_pixels * pixel_weight + np.array(scores_edges) * edge_weight
    best_idx = np.argmax(scores_final)
    best_x = x_positions[best_idx]
    
    # Convert grayscale to BGR (3-channel)
    img_with_bar = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Draw red bar using slice assignment (BGR format: red = (0, 0, 255))
    img_with_bar[:, best_x:best_x+width] = (0, 0, 255)
    
    # cv2.imshow("with_bar", img_with_bar)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return img_with_bar

from PIL import Image, ImageDraw, ImageFont

def wrap_text(text, font, max_width):
    dummy = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(dummy)

    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = word if current == "" else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return "\n".join(lines)

def text_to_img(text, font_path, font_size, out_path=None, padding=20, max_width=1200, line_spacing=8):
    font = ImageFont.truetype(str(font_path), font_size)
    wrapped = wrap_text(text, font, max_width)

    dummy = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=line_spacing)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img = Image.new("L", (text_w + 2 * padding, text_h + 2 * padding), 255)
    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (padding - bbox[0], padding - bbox[1]),
        wrapped,
        font=font,
        fill=0,
        spacing=line_spacing
    )

    if out_path is not None:
        img.save(out_path)

    return img

# def draw_hbar(img):
#     img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # assume img is grayscale
#     _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     ydata = []
#     letters = []

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         letters.append((x,y,w,h))
    
#     data = np.empty((len(letters), 1)) # lower and upper values
#     data[:,0] = [letter[1] for letter in letters]
#     #data[:,1] = [letter[1]+letter[3] for letter in letters]

#     clustering = DBSCAN(
#         eps=15,       # max vertical distance within line
#         min_samples=1
#     ).fit(data)

#     labels = clustering.labels_

#     lines = {}
#     for letter,label in zip(letters, labels):
#         if label in lines.keys():
#             lines[label].append(letter)
#         else:
#             lines[label] = [letter]

#     # # we care about the height of the letter
    
#     y_plus_h_data_per_line = []
#     y_data_per_line = []
#     h_data_per_line = []
#     for line in lines.values():
#         temp = []
#         temp2 = [] # for lower bound
#         temp3 = [] # for heights
#         for letter in line:
#             x,y,w,h = letter
#             temp.append(y)
#             temp2.append(y+h)
#             temp3.append(h)
#         y_plus_h_data_per_line.append(temp2)
#         y_data_per_line.append(temp)
#         h_data_per_line.append(temp3)

#     median_higher = [np.median(x) for x in y_data_per_line] # len: num_lines
#     median_height = [np.median(x) for x in h_data_per_line]

#     median_lower = [x+y for x,y in zip(median_higher, median_height)]

#     img_with_bar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     # Create a transparent overlay
#     overlay = img_with_bar.copy()  # Copy the original image

#     for lower, higher in zip(median_lower, median_higher):
#         higher = int(higher)
#         lower = int(lower)

#         # Draw a white rectangle on the overlay
#         cv2.rectangle(
#             overlay,
#             (0, higher),
#             (img.shape[1], lower),
#             (0, 255, 0),  # White color
#             thickness=-1
#         )

#     img = cv2.addWeighted(img_with_bar, 0.5, overlay, 0.5, 0)

#     cv2.imshow("overlay", img)
#     cv2.waitKey(0)

# def draw_hbar(img):
#     img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # assume img is grayscale
#     _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     ydata = []
#     letters = []

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         letters.append((x,y,w,h))
    
#     data = np.empty((len(letters), 1)) # all y-values
#     data[:,0] = [letter[1] for letter in letters]

#     clustering = DBSCAN(
#         eps=15,       # max vertical distance within line
#         min_samples=1
#     ).fit(data)

#     labels = clustering.labels_

#     lines = {}
#     for letter,label in zip(letters, labels):
#         if label in lines.keys():
#             lines[label].append(letter)
#         else:
#             lines[label] = [letter]

#     # # we care about the height of the letter
    
#     y_plus_h_data_per_line = []
#     y_data_per_line = []
#     for line in lines.values():
#         temp = []
#         temp2 = [] # for lower bound
#         for letter in line:
#             x,y,w,h = letter
#             temp.append(y)
#             temp2.append(y+h)
#         y_plus_h_data_per_line.append(temp2)
#         y_data_per_line.append(temp)

#     median_lower = [np.median(x) for x in y_plus_h_data_per_line] # len: num_lines
#     median_higher = [np.median(x) for x in y_data_per_line] # len: num_lines

#     img_with_bar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     # Create a transparent overlay
#     overlay = img_with_bar.copy()  # Copy the original image

#     for lower, higher in zip(median_lower, median_higher):
#         higher = int(higher)
#         lower = int(lower)

#         # Draw a white rectangle on the overlay
#         cv2.rectangle(
#             overlay,
#             (0, higher),
#             (img.shape[1], lower),
#             (0, 255, 0),  # White color
#             thickness=-1
#         )

#     img = cv2.addWeighted(img_with_bar, 0.5, overlay, 0.5, 0)

#     cv2.imshow("overlay", img)
#     cv2.waitKey(0)


def compute_scores(img,
                   show_contours=True,
                   show_boundaries=False,
                   showbar=True
                   ):
    img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height_total, width_total = img.shape

    # assume img is grayscale
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters = []
    if show_contours:
        cv2.drawContours(img, contours, contourIdx=-1, color=(0,0,255))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        letters.append((x,y,w,h))
    
    data = np.empty((len(letters), 1)) # lower and upper values
    data[:,0] = [letter[1] for letter in letters]
    print('yvalues:'); print(data[:,0])
    #sys.exit()

    clustering = DBSCAN(
        eps=15,       # max vertical distance within line
        min_samples=1
    ).fit(data)

    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('NUM CLUSTERS:', n_clusters)

    lines = {}
    for letter,label in zip(letters, labels):
        if label in lines.keys():
            lines[label].append(letter)
        else:
            lines[label] = [letter]

    # length of lines should be length of labels

    # # we care about the height of the letter
    
    y_plus_h_data_per_line = []
    y_data_per_line = []
    h_data_per_line = []
    for line in lines.values():
        temp = [] # upper bound
        temp2 = [] # for lower bound
        temp3 = [] # for heights
        for letter in line:
            x,y,w,h = letter
            temp.append(y)
            temp2.append(y+h)
            temp3.append(h)
            if show_boundaries:
                img = cv2.line(img, (0, y-1), (width_total, y+1), (0,255,255))
        y_plus_h_data_per_line.append(temp2)
        y_data_per_line.append(temp)
        h_data_per_line.append(temp3)

    if show_boundaries:
        cv2.imshow("img", img)
        cv2.waitKey(0)

    def differences(contours):
        leftmost_xs = []
        rightmost_xs = []
        for contour in contours:
            # contour is a numpy array of shape (N, 1, 2)
            xs = contour[:, 0, 0]  # All x-coordinates in the contour
            leftmost_x = np.min(xs)
            rightmost_x = np.max(xs)
            leftmost_xs.append(leftmost_x)
            rightmost_xs.append(rightmost_x)
        leftmost_xs.sort()
        rightmost_xs.sort()
        d = []
        for i in range(len(leftmost_xs)-1):
            diff = leftmost_xs[i+1] - rightmost_xs[i]
            d.append(diff)
        return d

    median_higher = [np.median(x) for x in y_data_per_line] # len: num_lines
    median_height = [np.median(x) for x in h_data_per_line]

    max_higher = [np.min(x) for x in y_data_per_line]
    max_height = [np.max(x) for x in h_data_per_line]

    median_lower = [x+y for x,y in zip(median_higher, median_height)]
    max_lower = [x+y for x,y in zip(max_higher, max_height)]

    stds_upper = []
    stds_lower = []
    meds_upper = []
    meds_lower = []
    # iterate over layers
    for med_h, med_l, max_h, max_l in zip(median_higher, median_lower, max_higher, max_lower):
        upper_section = img[int(max_h):int(med_h), :]
        lower_section = img[int(med_l):int(max_l), :]
        contours_upper, _ = cv2.findContours(upper_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_lower, _ = cv2.findContours(lower_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        diffs_upper = differences(contours_upper)
        diffs_lower = differences(contours_lower)
        if len(diffs_upper) > 0:
            stds_upper.append(np.std(diffs_upper))
            meds_upper.append(np.median(diffs_upper))
        else:
            stds_upper.append(None)
            meds_upper.append(None)
        if len(diffs_lower) > 0:
            stds_lower.append(np.std(diffs_lower))
            meds_lower.append(np.median(diffs_lower))
        else:
            stds_lower.append(None)
            meds_lower.append(None)

    # sort by layer
    idx = np.argsort(median_higher)
    sorted_stds_lower = np.array(stds_lower)[idx]
    sorted_stds_upper = np.array(stds_upper)[idx]
    sorted_meds_lower = np.array(meds_lower)[idx]
    sorted_meds_upper = np.array(meds_upper)[idx]
    # print('PRINTING STATS')
    # for i in idx:
    #     print('layer', i, ':')
    #     print('std lower:', sorted_stds_lower[i])
    #     print('std upper:', sorted_stds_upper[i])
    #     print('std meds:', sorted_meds_lower[i])
    #     print('std meds:', sorted_meds_upper[i])
    #     print()

    


    img_with_bar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Create a transparent overlay
    overlay = img_with_bar.copy()  # Copy the original image

    for lower, maxlow, higher, maxhigh in zip(median_lower, max_lower, median_higher, max_higher):
        higher = int(higher)
        lower = int(lower)
        maxlow = int(maxlow)
        mawhigh = int(maxhigh)

        # Draw a white rectangle on the overlay
        cv2.rectangle(
            overlay,
            (0, higher),
            (img.shape[1], lower),
            (0, 255, 0),
            thickness=-1
        )

        cv2.rectangle(
            overlay,
            (0, maxhigh),
            (img.shape[1], higher),
            (255, 0, 0),
            thickness=-1
        )

        cv2.rectangle(
            overlay,
            (0, lower),
            (img.shape[1], maxlow),
            (255, 0, 0),
            thickness=-1
        )

    if showbar:
        img = cv2.addWeighted(img_with_bar, 0.5, overlay, 0.5, 0)
        cv2.imshow("overlay", img)
        cv2.waitKey(0)

    return sorted_stds_lower, sorted_stds_upper, sorted_meds_lower, sorted_meds_upper


def compute_scores_att_2(img, call_idx, debug=False,
                         output_dir=Path.cwd() / "new_pipeline" / "outputs",
                         snippet_output_dir=Path.cwd() / "new_pipeline" / "snippet_outputs" # for debugging results
                         ):
    

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
        
    
    
    img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    height_total, width_total = img.shape

    # assume img is grayscale
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_with_bar = np.array(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    letters = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        letters.append((x,y,w,h))

    white_buffer = 2 # both directions (up and down)
    labels = np.full(height_total, np.nan)
    class_counter = 0
    in_white = True
    # None = white, int = class label
    buffer = 2
    for y in range(white_buffer, height_total - white_buffer):
        if np.all(img[y-buffer:y+buffer,:] == 255):
            in_white = True
        elif np.any(img[y,:] == 0):
            # there is black
            if in_white: # prev was white
                # new class
                if debug:
                    print('NEW CLASS')
                class_counter += 1
            labels[y] = class_counter
            in_white = False

    layerinfo = [] # stores, per layer, maxheight, medheight, medlow, maxlow
    if debug:
        print('DEBUG, number of layers:', class_counter)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cls in range(1, class_counter+1):
        valid_indices = np.where(labels == cls)[0]
        top = np.min(valid_indices)
        bottom = np.max(valid_indices)

        scores_top = np.full((bottom - top)//2, np.nan)
        scores_bottom = np.full((bottom - top)//2, np.nan)
        if len(scores_top) == 0:
            print('ERROR: with top and bottom:', top, bottom)
            # skip iteration
            
            img_copy = img.copy()
            cv2.rectangle(
                img_copy,
                (0, top), (width_total, bottom),
                (0, 255, 0),
                thickness=-1
            )
            cv2.imwrite(str(output_dir / f"failed_with_image_{call_idx}.png"), img_copy)
            return
        for i in range(len(scores_top)):
            intensity = np.sum(np.abs(img[i+top,:] - img[i+1+top,:]))
            scores_top[i] = intensity
            intensity = np.sum(np.abs(img[bottom-i,:] - img[bottom-i-1,:]))
            scores_bottom[i] = intensity

        padding_indices = 1 # to avoid having completely no margin
        
        best_top = np.argmax(scores_top) + top
        best_bottom = bottom - np.argmax(scores_bottom)

        # if abs(top - (best_top - padding_indices)) >= 1:
        #     best_top -= padding_indices
        # if abs(bottom - (best_bottom + padding_indices)) >= 1:
        #     best_bottom += padding_indices
        layerinfo.append((top, best_top, best_bottom, bottom))

    
    if debug:

        for x in layerinfo:
            t,bt,bb,b = x

            # Draw a white rectangle on the overlay
            cv2.rectangle(
                overlay,
                (0, bt),
                (width_total, bb),
                (0, 255, 0),
                thickness=-1
            )

            cv2.rectangle(
                overlay,
                (0, t),
                (img.shape[1], bt),
                (255, 0, 0),
                thickness=-1
            )

            cv2.rectangle(
                overlay,
                (0, bb),
                (img.shape[1], b),
                (255, 0, 0),
                thickness=-1
            )

        img_with_bar = cv2.addWeighted(img_with_bar, 0.5, overlay, 0.5, 0)
        

    upper_dists = []
    lower_dists = []

    contour_counter_upper = []
    contour_counter_lower = []

    for info in layerinfo:
        top, best_top, best_bottom, bottom = info
        if debug:
            print('DEBUG: top and besttop:', top, best_top)
            print('DEBUG: bot and bestbottom:', bottom, best_bottom)
        if best_top - top > padding_indices:
            best_top -= padding_indices
        if bottom - best_bottom > padding_indices:
            best_bottom += padding_indices
        upper_section = img[top:best_top,:]
        lower_section = img[best_bottom:bottom,:]

        try:
            #kernel = np.ones((2, 2), np.uint8)
            section = cv2.threshold(upper_section, 150, 255, cv2.THRESH_BINARY_INV)[1]
            #section = cv2.morphologyEx(section, cv2.MORPH_OPEN, kernel)
            contours_upper, _ = cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            section = cv2.threshold(lower_section, 150, 255, cv2.THRESH_BINARY_INV)[1]
            #section = cv2.morphologyEx(section, cv2.MORPH_OPEN, kernel)
            contours_lower, _ = cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            print('ERROR WHEN CUTTING:')
            print('top, best_top:', top, best_top)
            #cv2.imshow("img", img)
            #cv2.waitKey(0)
            return
        
        cv2.drawContours(
            img_with_bar,
            contours_upper,
            -1,
            (0, 0, 255),      # Red in BGR
            1,
            offset=(0, top)
        )
        cv2.drawContours(
            img_with_bar,
            contours_lower,
            -1,
            (0, 0, 255),      # Red in BGR
            1,
            offset=(0, best_bottom)
        )

        # contours_upper, _ = cv2.findContours(upper_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours_lower, _ = cv2.findContours(lower_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_counter_upper.append(len(contours_upper))
        contour_counter_lower.append(len(contours_lower))

        if debug:
            print('DEBUG: LEN CONTOURS UPPER:', len(contours_upper))
            print('DEBUG: LEN CONTOURS LOWER:', len(contours_lower))
        diffs_upper = differences(contours_upper)
        diffs_lower = differences(contours_lower)
        if debug:
            print('DEBUG: diffs upper:', len(diffs_upper))
            print('DEBUG: diffs lower:', len(diffs_lower))
        if len(diffs_upper) > 0:
            for x in diffs_upper:
                upper_dists.append(x)
        if len(diffs_lower) > 0:
            for x in diffs_lower:
                lower_dists.append(x)

    if debug:
        print('DEBUG: num contours per layer (lower):', contour_counter_lower)
        print('DEBUG: num contours per layer (upper):', contour_counter_upper)
        print('DEBUG DONE')
        print()

        cv2.imshow("overlay", img_with_bar)
        cv2.waitKey(0)
        cv2.imwrite(snippet_output_dir / "res.png", img_with_bar)

    return upper_dists, lower_dists
        
import textwrap

def compute_scores_att_3(img, call_idx, debug=False,
                         output_dir=Path.cwd() / "new_pipeline" / "outputs",
                         snippet_output_dir=Path.cwd() / "new_pipeline" / "snippet_outputs" # for debugging results
                         ):
    
    def differences(min_xs, max_xs):
        if len(min_xs) <= 1:
            return []

        paired = sorted(zip(min_xs, max_xs), key=lambda p: p[0])

        gaps = []
        for (_, right), (left, _) in zip(paired[:-1], paired[1:]):
            gaps.append(left - right)

        return gaps
    
    img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_with_bar = np.array(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) # in color
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    height_total, width_total = img.shape

    # assume img is grayscale
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_buffer = 2 # both directions (up and down)
    labels = np.full(height_total, np.nan)
    class_counter = 0
    in_white = True
    # None = white, int = class label
    buffer = 2
    for y in range(white_buffer, height_total - white_buffer):
        if np.all(img[y-buffer:y+buffer,:] == 255):
            in_white = True
        elif np.any(img[y,:] == 0):
            # there is black
            if in_white: # prev was white
                # new class
                if debug:
                    print('NEW CLASS')
                class_counter += 1
            labels[y] = class_counter
            in_white = False

    if debug:
        print('DEBUG, number of layers:', class_counter)

    all_upper_diffs = []
    all_lower_diffs = []

    for cls in range(1, class_counter+1):
        valid_indices = np.where(labels == cls)[0]
        top = np.min(valid_indices)
        bottom = np.max(valid_indices)
        section = img[top:bottom]
        section = cv2.threshold(section, 150, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights = []
        lows = []
        min_xvalues = []
        max_xvalues = []
        if len(contours) == 0:
            print('CLASS IDX:', cls)
            cv2.imwrite(snippet_output_dir / "res.png", img_with_bar)
            continue
        for contour in contours:
            ys = contour[:, 0, 1]
            xs = contour[:, 0, 0]
            heights.append(top + np.min(ys))
            lows.append(top + np.max(ys))
            min_xvalues.append(np.min(xs))
            max_xvalues.append(np.max(xs))
            if debug:
                print('DEBUG: appending min value:', min_xvalues[-1], 'and max value:', max_xvalues[-1])

        
        #print('semi debug: heights:', heights)
        med_h = int(np.median(heights))
        med_l = int(np.median(lows))
        y_thresh = 0 # min threshold to cross to be considered accender/decender
        min_xvalues_upper = [min_xvalues[i] for i in range(len(contours)) if heights[i] < med_h - y_thresh]
        min_xvalues_lower = [min_xvalues[i] for i in range(len(contours)) if lows[i] > med_l + y_thresh]
        max_xvalues_upper = [max_xvalues[i] for i in range(len(contours)) if heights[i] < med_h - y_thresh]
        max_xvalues_lower = [max_xvalues[i] for i in range(len(contours)) if lows[i] > med_l + y_thresh]
        if debug:
            contours_to_show = [contours[i] for i in range(len(contours)) if (
                heights[i] < med_h - y_thresh or
                lows[i] > med_l + y_thresh
            )]
            cv2.drawContours(img_with_bar, contours_to_show, contourIdx=-1, color=(0,0,255), offset=(0, top))
        if debug:
            print('DEBUG: all lengths:', len(min_xvalues_upper), len(min_xvalues_lower), len(max_xvalues_upper), len(max_xvalues_lower))
        assert len(min_xvalues_upper) == len(max_xvalues_upper), AssertionError(f"lengths upper: {len(min_xvalues_upper)} vs {len(max_xvalues_upper)}")
        assert len(min_xvalues_lower) == len(max_xvalues_lower), AssertionError(f"lengths lower: {len(min_xvalues_lower)} vs {len(max_xvalues_lower)}")

        upper_diffs = differences(min_xvalues_upper,
                          max_xvalues_upper)

        lower_diffs = differences(min_xvalues_lower,
                                max_xvalues_lower)
        
        for x in upper_diffs:
            all_upper_diffs.append(x)
        for x in lower_diffs:
            all_lower_diffs.append(x)
     
        if debug:
            print('DEBUG: drawing lines with med_h and med_l:', med_h, med_l)
            img_with_bar = cv2.line(img_with_bar, (0, med_h-1), (width_total, med_h+1), (0,255,0))
            img_with_bar = cv2.line(img_with_bar, (0, med_l-1), (width_total, med_l+1), (0,255,0))
    
    if debug:
        cv2.imshow("img_with_lines", img_with_bar)
        cv2.waitKey(0)
        cv2.imwrite(snippet_output_dir / "res.png", img_with_bar)

    return all_upper_diffs, all_lower_diffs
        
    
def wrap_text_to_pixels(text: str, font: ImageFont, max_width_px: int) -> list[str]:
    """
    Wrap text so each line is at most max_width_px pixels wide.
    Returns a list of lines (strings).
    """
    # First, wrap by characters (rough), then refine by pixel width
    rough_lines = textwrap.wrap(text, width=100)  # rough char-based wrap

    lines = []
    for rough_line in rough_lines:
        words = rough_line.split()
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = font.getbbox(test_line)
            test_width = bbox[2] - bbox[0]
            if test_width <= max_width_px:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                # Try to fit word alone
                while word:
                    bbox = font.getbbox(word)
                    w = bbox[2] - bbox[0]
                    if w <= max_width_px:
                        current_line = word
                        break
                    else:
                        # Split word
                        for i in range(len(word), 0, -1):
                            part = word[:i]
                            bbox = font.getbbox(part)
                            if bbox[2] - bbox[0] <= max_width_px:
                                lines.append(part)
                                word = word[i:]
                                break
                        else:
                            # Single char fallback
                            lines.append(word[0])
                            word = word[1:]
                else:
                    current_line = ""
        if current_line:
            lines.append(current_line)
    return lines


def make_text_images(
    text: str,
    font_path: str,
    font_size: int,
    max_width_px: int = 800,
    max_height_px: int = 1200,
    padding: int = 40,
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (0, 0, 0),
    center_horizontally: bool = True,
) -> list[Image.Image]:
    """
    Generate a list of images containing the text, splitting into multiple pages
    if the text is too long to fit in one image of max_height_px.

    - Font size is fixed; we only split into multiple images.
    - max_width_px: maximum line width in pixels.
    - max_height_px: maximum image height in pixels.
    - padding: padding around text on each side.
    """
    font = ImageFont.truetype(font_path, font_size)

    # Measure line height
    bbox_zero = font.getbbox("A")
    line_height = bbox_zero[3] - bbox_zero[1]
    # Add a small spacing
    line_height_with_spacing = line_height + 4

    # Wrap text to pixel width
    wrapped_lines = wrap_text_to_pixels(text, font, max_width_px - 2 * padding)

    if not wrapped_lines:
        return [Image.new("RGB", (max_width_px, max_height_px), bg_color)]

    # Compute height needed for all lines
    total_height_needed = len(wrapped_lines) * line_height_with_spacing
    total_width_needed = max_width_px  # we fix width

    # If fits in one page
    if total_height_needed <= max_height_px - 2 * padding:
        img = Image.new("RGB", (total_width_needed, total_height_needed + 2 * padding), bg_color)
        draw = ImageDraw.Draw(img)

        y = padding
        for line in wrapped_lines:
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            x = padding if not center_horizontally else (total_width_needed - line_w) // 2
            draw.text((x, y), line, font=font, fill=text_color)
            y += line_height_with_spacing
        return [img]

    # Otherwise, split into multiple pages
    max_lines_per_page = (max_height_px - 2 * padding) // line_height_with_spacing
    if max_lines_per_page <= 0:
        raise ValueError("max_height_px is too small for the given font and padding.")

    images = []
    for start in range(0, len(wrapped_lines), max_lines_per_page):
        page_lines = wrapped_lines[start:start + max_lines_per_page]
        page_height = len(page_lines) * line_height_with_spacing
        img = Image.new("RGB", (total_width_needed, page_height + 2 * padding), bg_color)
        draw = ImageDraw.Draw(img)

        y = padding
        for line in page_lines:
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            x = padding if not center_horizontally else (total_width_needed - line_w) // 2
            draw.text((x, y), line, font=font, fill=text_color)
            y += line_height_with_spacing
        images.append(img)

    return images