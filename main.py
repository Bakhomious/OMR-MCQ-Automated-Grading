import cv2 as cv
import imutils
import numpy as np
from imutils.perspective import four_point_transform

HEIGHT = 500
WIDTH = 300
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

image = cv.imread("images/test_17.png")
image = cv.resize(image, (WIDTH, HEIGHT))
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(image, (5, 5), 0)
edges = cv.Canny(blur, 10, 70)

# cv.imwrite("edges.png", edges)
contours = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]

# cv.drawContours(image, contours, -1, GREEN, 2)
# cv.imwrite("contours.png", image)

def get_rect_cnts(contours):
    rect_cnts = []
    for cnt in contours:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            rect_cnts.append(approx)

    rect_cnts = sorted(rect_cnts, key=cv.contourArea, reverse=True)

    return rect_cnts

rect_cnts = get_rect_cnts(contours)
document = four_point_transform(image, rect_cnts[0].reshape(4, 2))

cv.drawContours(image, rect_cnts, -1, (255, 0, 0), 3)

# cv.imwrite("rectangles.png", image)
# cv.imwrite("birdeye.png", document)

gray_doc = cv.cvtColor(document, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray_doc, 170, 255, cv.THRESH_BINARY_INV)[1]

# cv.imwrite("threshold.png", thresh)

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    bounding_boxes = [cv.boundingRect(c) for c in cnts]
    cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Find Number of Rows and columns (if wanted to add more questions)
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts_horizontal = cv.findContours(horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts_horizontal = cnts_horizontal[0] if len(cnts_horizontal) == 2 else cnts_horizontal[1]
rows = 0
cnts_horizontal = sort_contours(cnts_horizontal, method="top-to-bottom")
for i, c in enumerate(cnts_horizontal):
    #if i == 1:  # nth + 1 horizontal line
    #   cv.drawContours(document, [c], -1, (36,255,12), 2)
    rows += 1

vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,25))
vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=2)
cnts_vertical = cv.findContours(vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts_vertical = cnts_vertical[0] if len(cnts_vertical) == 2 else cnts_vertical[1]
columns = 0
cnts_vertical = sort_contours(cnts_vertical)
for i, c in enumerate(cnts_vertical):
    #if i == 1:  # nth + 1 vertical line
    #    cv.drawContours(document, [c], -1, (36, 255, 12), 2)
    columns += 1

print(f'{rows - 1}, {columns - 1}')

cv.imwrite("document.png", document)

def get_cell(row, column):
    # Get the bounding rectangles of the specified horizontal and vertical contours
    h1_x, h1_y, h1_w, h1_h = cv.boundingRect(cnts_horizontal[row - 1])
    h2_x, h2_y, h2_w, h2_h = cv.boundingRect(cnts_horizontal[row])
    v1_x, v1_y, v1_w, v1_h = cv.boundingRect(cnts_vertical[column - 1])
    v2_x, v2_y, v2_w, v2_h = cv.boundingRect(cnts_vertical[column])

    # Calculate the coordinates of the region of interest (ROI)
    x = v1_x + v1_w
    y = h1_y + h1_h
    w = v2_x - x
    h = h2_y - y

    # Extract the ROI from the document
    cell = document[y:y + h, x:x + w]

    return cell

roi = get_cell(3, 3)
cv.imwrite("roi.png", roi)

cv.waitKey()