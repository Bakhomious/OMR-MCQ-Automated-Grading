import csv

import cv2 as cv
from imutils.perspective import four_point_transform

from main import *
from constants import FOLDER_PATH, WIDTH, HEIGHT


def extract_answers(document, horizontal_contours, vertical_contours):
    list_of_rows = []
    for row in range(2, len(horizontal_contours)):
        row_line = []
        for col in range(2, len(vertical_contours)):
            roi = get_cell(document, row, col, horizontal_contours, vertical_contours)
            if is_cross_present(roi):
                row_line.append(1)
            else:
                row_line.append(0)
        list_of_rows.append(row_line)
    return list_of_rows


def extract_and_read():

    image_int = cv.imread(f'{FOLDER_PATH}\\test_solver.png')
    image, edges = preprocess_image(image_int, WIDTH, HEIGHT)
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    rect_cnts = get_rectangular_contours(contours)
    document = four_point_transform(image, rect_cnts[0].reshape(4, 2))
    thresh = get_thresh(document, 125)

    horizontal_contours, vertical_contours = get_grid_contours(thresh)

    # Improve accuracy
    cv.drawContours(document, vertical_contours, -1, GREEN, 2)

    print(len(horizontal_contours), len(horizontal_contours))

    answers = extract_answers(document, horizontal_contours, vertical_contours)

    with open("answer_key_test.csv", mode="w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        for answer in answers:
            writer.writerow(answer)

    return read_answer_key("answer_key_test.csv")