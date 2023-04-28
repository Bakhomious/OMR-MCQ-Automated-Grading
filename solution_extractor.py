from main import *
from constants import FOLDER_PATH, WIDTH, HEIGHT


def extract_answers(document, horizontal_contours, vertical_contours):
    """
    Extract the answers from the OMR sheet by iterating over all the cells in the grid and checking if a cross is present.

    Args:
        document (ndarray): The image of the OMR sheet.
        horizontal_contours (list): The horizontal contours of the grid.
        vertical_contours (list): The vertical contours of the grid.

    Returns:
        list: A 2D list where each row represents a row of the grid and contains 1s and 0s indicating whether a cross is present in that cell or not.
    """
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
    """
    extract_and_read() function reads the answer key from the test_solver.png image, extracts the answers and saves them in a CSV file, and returns a dictionary of the answers.

    Returns:
    - answer_key: A dictionary that contains the answers from the image.

    Steps:
    1. Reads the test_solver.png image using OpenCV.
    2. Preprocesses the image and gets its contours.
    3. Gets the rectangular contours from the image.
    4. Applies four-point transformation to the image to obtain a top-down view.
    5. Applies thresholding to the image to get a binary image.
    6. Gets the horizontal and vertical contours of the grid in the image.
    7. Extracts the answers from the image and saves them in a CSV file.
    8. Reads the CSV file and returns a dictionary that contains the answers.
    """
    image_int = cv.imread(f'{FOLDER_PATH}\\test_solver.png')
    image, edges = preprocess_image(image_int, WIDTH, HEIGHT)
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    rect_cnts = get_rectangular_contours(contours)
    document = four_point_transform(image, rect_cnts[0].reshape(4, 2))
    thresh = get_thresh(document, 125)

    horizontal_contours, vertical_contours = get_grid_contours(thresh)

    # Improve accuracy
    cv.drawContours(document, vertical_contours, -1, GREEN, 2)

    answers = extract_answers(document, horizontal_contours, vertical_contours)

    with open("answer_key_test.csv", mode="w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        for answer in answers:
            writer.writerow(answer)

    return read_answer_key("answer_key_test.csv")