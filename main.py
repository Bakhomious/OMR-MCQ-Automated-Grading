import operator

import cv2 as cv
import numpy as np
from imutils.perspective import four_point_transform
import csv
import os
import re
import solution_extractor

FOLDER_PATH = "images"

# Constants for resizing
HEIGHT = 500
WIDTH = 300

# Colors for contour drawing
GREEN = (0, 255, 0)
RED = (0, 0, 255)


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


def get_cell(document, row, column, cnts_horizontal, cnts_vertical):
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


def get_thresh(document, lb=170):
    gray_doc = cv.cvtColor(document, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray_doc, lb, 255, cv.THRESH_BINARY_INV)[1]
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


# Read the answer key from a csv file
def read_answer_key(file_path):
    answer_key = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            answer_key.append(row)
    return answer_key


# Check if a cross is present in the ROI
def is_cross_present(roi, threshold=0.15):
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, thresh_roi = cv.threshold(gray_roi, 128, 255, cv.THRESH_BINARY_INV)
    # cv.medianBlur(thresh_roi, 5, thresh_roi)
    cv.dilate(thresh_roi, np.ones((20, 20), np.uint8), thresh_roi, iterations=1)

    white_pixels = np.sum(thresh_roi == 255)
    total_pixels = thresh_roi.size
    if white_pixels / total_pixels > threshold:
        return True
    return False


# Iterate over the ROIs, skipping the first row and column, and compare the extracted answers with the answer key
def grade(document, answer_key, horizontal_contours, vertical_contours):
    correct_answers = 0
    empty_rows = 0
    wrong_answers = 0
    for row in range(2, len(horizontal_contours)):
        crosses_in_row = 0
        correct_answer_col = -1
        for col in range(2, len(vertical_contours)):
            roi = get_cell(document, row, col, horizontal_contours, vertical_contours)
            if is_cross_present(roi):
                crosses_in_row += 1
                if answer_key[row - 2][col - 2] == "1":
                    correct_answer_col = col

        if crosses_in_row == 0:
            empty_rows += 1
        elif crosses_in_row == 1 and correct_answer_col != -1:
            correct_answers += 1
        else:
            wrong_answers += 1

    return correct_answers, empty_rows, wrong_answers


if __name__ == '__main__':
    results_list = []
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".png") and not filename.startswith("test_solver"):
            image_int = cv.imread(f'{FOLDER_PATH}\{filename}')
            image, edges = preprocessing(image_int)
            contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

            rect_cnts = get_rect_cnts(contours)
            document = four_point_transform(image, rect_cnts[0].reshape(4, 2))
            thresh = get_thresh(document)

            horizontal_contours, vertical_contours = get_grid_contours(thresh)

            answer_key = solution_extractor.extract_and_read()
            correct_answers, empty_rows, wrong_answers = grade(document, answer_key, horizontal_contours, vertical_contours)

            total_questions_answered = len(horizontal_contours) - empty_rows - 2
            total_questions = len(horizontal_contours) - 2
            percentage = (correct_answers / total_questions) * 100

            print("File Name:", filename)
            print("Total Questions answered:", total_questions_answered)
            print("Wrong answers:", wrong_answers)
            print("Correct answers:", correct_answers)
            print("Empty rows:", empty_rows)
            print(f"Percentage: {percentage}%")

            results_dict = {"Image ID": int(re.findall(r'\d+', filename)[0]),
                            "Total Questions Answered": total_questions_answered,
                            "Correct": correct_answers,
                            "Wrong": wrong_answers,
                            "Empty": empty_rows,
                            "Percentage": percentage}
            results_list.append(results_dict)
            results_list.sort(key=operator.itemgetter("Image ID"))

            with open("results.csv", mode="w", newline='') as csv_file:
                fieldnames = ["Image ID", "Total Questions Answered", "Correct", "Wrong",
                              "Empty", "Percentage"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for result in results_list:
                    writer.writerow(result)