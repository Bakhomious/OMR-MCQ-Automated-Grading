import cv2 as cv
import imutils
import numpy as np
from imutils.perspective import four_point_transform

HEIGHT = 500
WIDTH = 300
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

image = cv.imread("images/test_0.png")
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

# Find Number of Rows and columns (if wanted to add more questions)
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv.findContours(horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
rows = 0

for c in cnts:
    # cv.drawContours(document, [c], -1, (36,255,12), 2)
    rows += 1

vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,25))
vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv.findContours(vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
columns = 0
for c in cnts:
    # cv.drawContours(document, [c], -1, (36,255,12), 2)
    columns += 1

print(f'{rows - 1}, {columns - 1}')

cv.waitKey()