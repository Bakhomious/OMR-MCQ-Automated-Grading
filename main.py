import csv
import operator
import os
import re

import cv2 as cv
import numpy as np
from imutils.perspective import four_point_transform

import solution_extractor
from constants import FOLDER_PATH, HEIGHT, WIDTH, GREEN, RED, BLUE, YELLOW


def preprocess_image(src_image, width, height):
    """
    Preprocesses the input image by resizing it, converting it to grayscale,
    applying Gaussian blur, and detecting edges using the Canny algorithm.

    Args:
        src_image (numpy.ndarray): The input image.
        width (int): The width to resize the image.
        height (int): The height to resize the image.

    Returns:
        numpy.ndarray: The resized image.
        numpy.ndarray: The edge-detected image.
    """
    resized_image = cv.resize(src_image, (width, height))
    gray_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    edge_image = cv.Canny(blurred_image, 10, 70)

    return resized_image, edge_image


def get_rectangular_contours(contours):
    """
    Filters the input contours by selecting only those with four points (rectangular shape),
    and sorts them in descending order by their contour area.

    Args:
        contours (list): A list of contours to filter and sort.

    Returns:
        list: A list of rectangular contours sorted by contour area in descending order.
        """
    rectangular_contours = []
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            rectangular_contours.append(approx)

    rectangular_contours = sorted(rectangular_contours, key=cv.contourArea, reverse=True)

    return rectangular_contours


def sort_contours(contours, method="left-to-right"):
    """
    Sorts the input contours based on the specified sorting method.

    Args:
        contours (list): A list of contours to be sorted.
        method (str, optional): The sorting method to be used.
                                Options: "left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top".
                                Defaults to "left-to-right".

    Returns:
        list: A list of sorted contours.
    """
    reverse = method in ["right-to-left", "bottom-to-top"]
    index = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0

    bounding_boxes = [cv.boundingRect(c) for c in contours]
    contours, _ = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][index], reverse=reverse))

    return contours


def get_cell(document, row, column, horizontal_contours, vertical_contours):
    """
    Extracts the cell (region of interest) from the document image based on row and column indices.

    Args:
        document (numpy.ndarray): The document image.
        row (int): The row index of the cell.
        column (int): The column index of the cell.
        horizontal_contours (list): The list of horizontal contours.
        vertical_contours (list): The list of vertical contours.

    Returns:
        numpy.ndarray: The extracted cell image.
    """
    # Get the bounding rectangles of the specified horizontal and vertical contours
    h1_x, h1_y, h1_w, h1_h = cv.boundingRect(horizontal_contours[row - 1])
    h2_x, h2_y, h2_w, h2_h = cv.boundingRect(horizontal_contours[row])
    v1_x, v1_y, v1_w, v1_h = cv.boundingRect(vertical_contours[column - 1])
    v2_x, v2_y, v2_w, v2_h = cv.boundingRect(vertical_contours[column])

    # Calculate the coordinates of the region of interest (ROI)
    x = v1_x + v1_w
    y = h1_y + h1_h
    w = v2_x - x
    h = h2_y - y

    # Extract the ROI from the document
    cell = document[y:y + h, x:x + w]

    return cell


def get_thresh(document, lower_bound=170):
    """
    Applies a binary inverse threshold to the input document image.

    Args:
        document (numpy.ndarray): The document image.
        lower_bound (int, optional): The lower bound value for the threshold. Defaults to 170.

    Returns:
        numpy.ndarray: The thresholded image.
    """
    gray_doc = cv.cvtColor(document, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_doc, lower_bound, 255, cv.THRESH_BINARY_INV)

    return thresh

def get_grid_contours(thresh):
    """
    Extracts horizontal and vertical contours from the thresholded image.

    Args:
        thresh (numpy.ndarray): The thresholded image.

    Returns:
        tuple: A tuple containing two lists: horizontal contours and vertical contours.
    """
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


def read_answer_key(file_path):
    """
    Reads the answer key from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the answer key.

    Returns:
        list: A list of lists, where each inner list represents a row of answers from the CSV file.
    """
    answer_key = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            answer_key.append(row)
    return answer_key


def is_cross_present(roi, threshold=0.15):
    """
    Checks if a cross is present in the region of interest (ROI).

    Args:
        roi (numpy.ndarray): The region of interest.
        threshold (float, optional): The threshold for the ratio of white pixels to total pixels. Defaults to 0.15.

    Returns:
        bool: True if a cross is present in the ROI, otherwise False.
    """
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, thresh_roi = cv.threshold(gray_roi, 128, 255, cv.THRESH_BINARY_INV)
    cv.dilate(thresh_roi, np.ones((20, 20), np.uint8), thresh_roi, iterations=1)

    white_pixels = np.sum(thresh_roi == 255)
    total_pixels = thresh_roi.size
    if white_pixels / total_pixels > threshold:
        return True
    return False


def grade(document, answer_key, horizontal_contours, vertical_contours):
    """
    Grades the exam document by comparing the extracted answers with the answer key.

    Args:
        document (numpy.ndarray): The transformed exam document.
        answer_key (list): The list of correct answers.
        horizontal_contours (list): The list of horizontal grid contours.
        vertical_contours (list): The list of vertical grid contours.

    Returns:
        int: The number of correct answers.
        int: The number of empty rows.
        int: The number of wrong answers.
    """
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


def process_image(filename):
    """
    Process the given image file and returns the document and threshold.

    Args:
        filename (str): The name of the image file to process.

    Returns:
        tuple: A tuple containing the following elements:
            - document (numpy.ndarray): The transformed and preprocessed document image.
            - thresh (numpy.ndarray): The thresholded image used for further analysis.
    """
    image_int = cv.imread(f'{FOLDER_PATH}/{filename}')
    image, edges = preprocess_image(image_int, WIDTH, HEIGHT)
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    rect_cnts = get_rectangular_contours(contours)
    document = four_point_transform(image, rect_cnts[0].reshape(4, 2))
    thresh = get_thresh(document)

    return document, thresh


def save_results(results_list, output_file="results.csv"):
    """
    Save the grading results to a CSV file.

    Args:
        results_list (list): A list of dictionaries containing grading results for each image.
        output_file (str, optional): The name of the output CSV file. Defaults to "results.csv".

    Returns:
        None
    """
    with open(output_file, mode="w", newline='') as csv_file:
        fieldnames = ["Image ID", "Total Questions Answered", "Correct", "Wrong", "Empty", "Percentage"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results_list:
            writer.writerow(result)


def main():
    results_list = []

    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".png") and not filename.startswith("test_solver"):
            document, thresh = process_image(filename)
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

    save_results(results_list)


if __name__ == '__main__':
    main()