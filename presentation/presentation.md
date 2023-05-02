---
marp: true
theme: gradient
class: blue
paginate: true
size: 4:3
---

<style>
    img {
        max-width: 200px;
    }
    .side-by-side {
        display: flex;
        justify-content: space-evenly;
        align-items: center;
        text-align: center;
    }
    table{
        font-size: 14pt;
    }
    span{
        font-style: italic;
        font-size: 14pt;
    }
</style>

# OMR MCQ Automated Grading System

---

## Introduction

- Simplifies and streamlines MCQ grading process
    - Uses image processing techniques and computer vision algorithms
- Automatically detects and extracts MCQ grids from scanned answer sheets
    - Recognizes and marks correct and incorrect answers
- Handles multiple crosses and empty rows
- Saves time, reduces human errors, and increases efficiency

---
## Features
1. Automatic detection and extraction of MCQ grids

<div class="side-by-side">
  <div>
    <img src="images/test_0.png" alt="Original">
    <br>
    <span>Original</span>
  </div>
  <div>
    <img src="images/detecttable.png" alt="Detected Grid">
    <br>
    <span>Detected Grid</span>
  </div>
</div>

---
## Features
2. Recognition and marking of correct and incorrect answers

<div class="side-by-side">
  <div>
    <img src="images/test_0.png" alt="Original">
    <br>
    <span>Original</span>
  </div>
  <div>
    <img src="images/checked_0.png" alt="Checked">
    <br>
    <span>Checked</span>
  </div>
</div>

---
## Features
3. Handling multiple crosses and empty rows

<div class="side-by-side">
  <div>
    <img src="images/checked_10.png" alt="Multiple crosses">
    <br>
    <span>Multiple Crosses</span>
  </div>
  <div>
    <img src="images/checked_16.png" alt="Empty row">
    <br>
    <span>Empty Row</span>
  </div>
</div>


---
## Features
4. Exporting results as marked images and `.csv`

<div class="side-by-side">

| Image ID | Total Questions Answered | Correct | Wrong | Empty | Percentage |
| -------- | ------------------------ | ------- | ----- | ----- | ---------- |
| 0        | 10                       | 7       | 3     | 0     | 70.0       |
| 1        | 10                       | 2       | 8     | 0     | 20.0       |
| 2        | 10                       | 3       | 7     | 0     | 30.0       |
| 3        | 10                       | 4       | 6     | 0     | 40.0       |
| 4        | 8                        | 2       | 6     | 2     | 20.0       |
| 5        | 10                       | 2       | 8     | 0     | 20.0       |
| 6        | 8                        | 2       | 6     | 2     | 20.0       |
| 7        | 10                       | 5       | 5     | 0     | 50.0       |
| 8        | 10                       | 2       | 8     | 0     | 20.0       |
| 9        | 10                       | 3       | 7     | 0     | 30.0       |
| 10       | ...                      | ...     | ...   | ...   | ...        |

</div>

---
## Workflow
### 1. Image preprocessing

- Grayscale conversion
    - `cv.cvtColor(image, cv.COLOR_BGR2GRAY)`
- Thresholding
    - `cv.threshold(gray_doc, lower_bound, 255, cv.THRESH_BINARY_INV)`
- Contour detection
    - `cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)`
    - Filtering horizontal and vertical contours

---
## Workflow
### 1. Image preprocessing
<div class="side-by-side">
    <div>
        <img src="images/edges.png" alt="Preprocessed">
        <br>
        <span>Preprocessed</span>
    </div>
</div>

---
## Workflow
### 2. Cell extraction from the Grid

- Looping through contours and calculating the bounding rectangle
- Creating the grid using horizontal and vertical contours
- Extracting cells using contour coordinates

<div class="side-by-side">
    <div>
        <img src="images/roi.png" alt="Extracted Cell" width="100">
        <br>
        <span>Extracted Cell</span>
    </div>
</div>

---
## Workflow
### 3. Answer marking
- Cross detection using threshold and ratio of white to black pixels
- Coloring cells based on answer correctness
  - Green: Correct
  - Red: Incorrect
  - Blue: Multiple crosses

---

## Demo

---

Thank you for your attention!
