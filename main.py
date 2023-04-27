import cv2 as cv
import imutils
import numpy as np
from imutils.perspective import four_point_transform
import csv

HEIGHT = 500
WIDTH = 300
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def preprocessing(image):
    image = cv.resize(image, (WIDTH, HEIGHT))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 10, 70)



    return image, edges


def get_rect_cnts(contours):
    rect_cnts = []
    for cnt in contours:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            rect_cnts.append(approx)

    rect_cnts = sorted(rect_cnts, key=cv.contourArea, reverse=True)

    return rect_cnts


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


def get_cell(row, column, cnts_horizontal, cnts_vertical):
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


def get_thresh(document):
    gray_doc = cv.cvtColor(document, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray_doc, 170, 255, cv.THRESH_BINARY_INV)[1]
    return thresh


def get_grid_contours(thresh):
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
    horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts_horizontal = cv.findContours(horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_horizontal = cnts_horizontal[0] if len(cnts_horizontal) == 2 else cnts_horizontal[1]
    cnts_horizontal = sort_contours(cnts_horizontal, method="top-to-bottom")

    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 25))
    vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts_vertical = cv.findContours(vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_vertical = cnts_vertical[0] if len(cnts_vertical) == 2 else cnts_vertical[1]
    cnts_vertical = sort_contours(cnts_vertical)

    return cnts_horizontal, cnts_vertical

    # Step 1: Add a function to read the answer key from a csv file
def read_answer_key(file_path):
    answer_key = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            answer_key.append(row)
    return answer_key

    # Step 2: Add a function to check if a cross is present in the ROI
def is_cross_present(roi, threshold=0.15):
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    # cv.imwrite("gray_roi.png", gray_roi) # For debugging purposes
    _, thresh_roi = cv.threshold(gray_roi, 128, 255, cv.THRESH_BINARY_INV)
    #cv.medianBlur(thresh_roi, 5, thresh_roi)
    cv.dilate(thresh_roi, np.ones((5, 5), np.uint8), thresh_roi, iterations=1)

    # cv.imwrite("thresh_roi.png", thresh_roi) # For debugging purposes

    white_pixels = np.sum(thresh_roi == 255)
    total_pixels = thresh_roi.size
    print(white_pixels / total_pixels)
    if white_pixels / total_pixels > threshold:
        return True
    return False


if __name__ == '__main__':
    image_int = cv.imread("images/test_9.png")
    image, edges = preprocessing(image_int)
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    rect_cnts = get_rect_cnts(contours)
    cv.imwrite("rect_cnts.png", cv.drawContours(image.copy(), rect_cnts, -1, (0, 255, 0), 3))
    document = four_point_transform(image, rect_cnts[0].reshape(4, 2))
    # document = cv.copyMakeBorder(document, 3, 3, 3, 3, cv.BORDER_CONSTANT, value=0)
    cv.imwrite("document.png", document)
    thresh = get_thresh(document)

    horizontal_contours, vertical_contours = get_grid_contours(thresh)

    #roi = get_cell(3, 3, horizontal_contours, vertical_contours)
    #cv.imwrite("roi.png", roi)

    answer_key = read_answer_key("answer_key_2.csv")

    # Step 3: Iterate over the ROIs, skipping the first row and column, and compare the extracted answers with the answer key
    correct_answers = 0
    total_answers = 0
    empty_rows = 0
    print(len(horizontal_contours), len(vertical_contours))
    wrong_answers = 0
    for row in range(2, len(horizontal_contours)):
        crosses_in_row = 0
        correct_answer_col = -1
        for col in range(2, len(vertical_contours)):
            roi = get_cell(row, col, horizontal_contours, vertical_contours)
            if is_cross_present(roi):
                crosses_in_row += 1
                if answer_key[row - 2][col - 2] == "1":
                    correct_answer_col = col
                total_answers += 1

        if crosses_in_row == 0:
            empty_rows += 1
        elif crosses_in_row == 1 and correct_answer_col != -1:
            correct_answers += 1
        else:
            wrong_answers += 1

    # roi = get_cell(4, 3, horizontal_contours, vertical_contours) # For debugging purposes
    # cv.imwrite("roi.png", roi) # For debugging purposes
    # print(is_cross_present(roi)) # For debugging purposes

    Total_answers = len(horizontal_contours) - empty_rows - 2
    print("Total Questions answered:", Total_answers)
    print("Wrong answers:", wrong_answers)
    print("Correct answers:", correct_answers)
    print("Empty rows:", empty_rows)
    #print("Accuracy:", correct_answers / total_answers * 100)

    cv.waitKey()
