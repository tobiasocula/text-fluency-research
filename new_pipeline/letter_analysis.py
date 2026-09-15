import numpy as np
import sys

# UPPERS ARE MUTUALLY EXCLUSIVE
upperleft = "bhk"
upperright = "d"
upperboth = "fijltäëöàáéíóúüñőűåèï"
caps = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

normal = "acemnorsuvwxz"

# LOWERS ARE MUTUALLY EXCLUSIVE
lowerleft = "py"
lowerright = "q"
lowerboth = "gjç"

double_accent = "äëöüőűï"

def analyze_text(text):

    upper_distances = []
    lower_distances = []

    count_lower_dist = 0
    count_upper_dist = 0

    for x in text:

        if x in caps:
            lower_distances.append(count_lower_dist)
            count_lower_dist = 0
        else:

            # LOWER
            if x in lowerleft:
                lower_distances.append(count_lower_dist)
                count_lower_dist = 0.5
            elif x in lowerright:
                lower_distances.append(count_lower_dist + 0.5)
                count_lower_dist = 0
            elif x in lowerboth:
                lower_distances.append(count_lower_dist)
                count_lower_dist = 0
            else:
                count_lower_dist += 1

            # UPPER
            if x in upperleft:
                upper_distances.append(count_upper_dist)
                count_upper_dist = 0.5
            elif x in upperright:
                upper_distances.append(count_upper_dist + 0.5)
                count_upper_dist = 0
            elif x in upperboth:
                upper_distances.append(count_upper_dist)
                count_upper_dist = 0
                if x in double_accent:
                    upper_distances.append(0.2) # small distance
            else:
                count_upper_dist += 1

    return upper_distances, lower_distances

        