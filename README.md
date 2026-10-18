Small research project I have/had going on with my university.

## Goals

#### Overview
Quantify the rhythm of texts, based on the language and font.

#### Horizontal rhythm

Given a text, we want to extract the "irregularities" that break up the horizontal rhythm, meaning measure fluency by looking at the "tall" and "lower" parts of each line of text. For this, we arrange a separation line which isolates the tall and lower parts of each letter, and derive statistics from both the lower and upper part of the line of text.

<!-- ![Hor-1](assets/hor_rhythm_example_2.png)

![Hor-1](assets/horizontal_rhythm_example.png) -->

<p align="center">
  <img src="assets/hor_rhythm_example_2.png" width="45%" />
  <img src="assets/horizontal_rhythm_example.png" width="45%" />
</p>

#### Vertical rhythm

###### (still ongoing)

Given a text, we want to measure the regularities of the "vertical rhythm" of the text. For this, we assign each letter with a custom bar profile, where one would imagine the most "vertical mass" to be present. Given the text, one then uses something like optical character recognition to draw the bars at correct positions and derives statistics from this.

<p align="center">
  <img src="new_vertical/letters_w_bars/A_cap.png" width="10%" />
  <img src="new_vertical/letters_w_bars/Z.png" width="10%" />
</p>