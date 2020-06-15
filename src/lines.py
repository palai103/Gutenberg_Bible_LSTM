import json
import operator
import os
import pathlib
import pickle
import shutil
from typing import List, Tuple

import cv2
import numpy as np
from tensorflow.keras.preprocessing import sequence

from numeric_order_corrector import rearrange_numeration

LABELS_DICTIONARY = {0: "-"}
max_label_length = 0
number_equivalent = 1


def save_json(filename, data):
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)


def pad_data(array, max_width, max_height, max_channels):
    padded_array = []

    for img_array in array:
        img_height = img_array.shape[0]
        img_width = img_array.shape[1]
        img_channels = img_array.shape[2]

        img_array_pad = np.pad(img_array,
                               pad_width=((0, max_height - img_height),
                                          (0, max_width - img_width),
                                          (0, max_channels - img_channels)),
                               constant_values=0.0)
        padded_array.append(img_array_pad)

    return np.array(padded_array)


def save_pickle(array, filename):
    outfile = open(filename, 'wb')
    pickle.dump(array, outfile)
    outfile.close()


def binarize_image(image_data):
    if len(image_data.shape) == 3:
        gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        binary_image = image_data
    return binary_image


def detect_lines(image_path) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """
    Detect text lines on a page image
    :param image_path: path to image where to detect lines
    :return: a tuple `(column_indicators, rows_indicators)` where:
     * `column_indicators` (list) contains x-axis boundary coordinates of the two biggest columns
     * `row_indicators` (list of lists) contains a list of y-axis lists of coordinates (one list for each column)
     Such that for each column `col`, `column_indicators[col] == (left, right)`
     and `row_indicators[col][i]` is the top of the i'th row
    """
    COLUMN_HISTOGRAM_THRESHOLD = 15
    ROW_HISTOGRAM_THRESHOLD = 50
    ROW_PROXIMITY_THRESHOLD = 10
    FIRST_ROW_SEPARATION_OFFSET = 25
    COLUMN_EXTRA_MARGIN = 14
    ROW_MARGIN_OFFSET = 5
    LINES_IN_PAGE = 42

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    binary_image = binarize_image(image)

    columns_histogram = cv2.reduce(binary_image, 0, cv2.REDUCE_AVG).reshape(-1)
    columns_histogram = np.array([1 if x > COLUMN_HISTOGRAM_THRESHOLD else 0 for x in columns_histogram])

    columns_indicators = []
    for column, _ in enumerate(columns_histogram[:-1]):
        if abs(columns_histogram[column] - columns_histogram[column + 1]) == 1:
            columns_indicators.append(column)

    """
    Add an extra margin of COLUMN_EXTRA_MARGIN pixels to the right of the two biggest columns
    """
    column_widths = {}
    for column_index in range(len(columns_indicators[:-1])):
        column_widths[(column_index, column_index + 1)] = abs(
            columns_indicators[column_index] - columns_indicators[column_index + 1])

    (first_left, first_right), (second_left, second_right) = sorted(list(
        sorted(column_widths, key=column_widths.get, reverse=True))[:2], key=operator.itemgetter(0))
    columns_indicators[first_right] += COLUMN_EXTRA_MARGIN
    columns_indicators[second_right] += COLUMN_EXTRA_MARGIN

    columns = []
    for (indicator, _) in enumerate(columns_indicators[:-1]):
        columns.append(binary_image[:, columns_indicators[indicator]:columns_indicators[indicator + 1]])

    rows_histograms = np.zeros((1, height))
    for column in columns:
        rows_histogram_column = cv2.reduce(column, 1, cv2.REDUCE_AVG)
        rows_histogram_column = np.array(
            [1 if x > ROW_HISTOGRAM_THRESHOLD else 0 for x in rows_histogram_column]).reshape((1, -1))
        rows_histograms = np.append(rows_histograms, rows_histogram_column, axis=0)
    rows_histograms = np.delete(rows_histograms, [0], axis=0)

    rows_indicators = []
    for row_indicator in rows_histograms:
        row_section_indicators = []
        for row, _ in enumerate(row_indicator[:-1]):
            if (row_indicator[row] - row_indicator[row + 1]) == 1 and (
                    not row_section_indicators or abs(row_section_indicators[-1] - row) > ROW_PROXIMITY_THRESHOLD):
                row_section_indicators.append(row + ROW_MARGIN_OFFSET)

        if row_section_indicators:  # `row_section_indicators` may be empty in the case of white columns
            row_section_indicators = row_section_indicators[-LINES_IN_PAGE:]
            row_section_indicators.insert(0, row_section_indicators[0] - FIRST_ROW_SEPARATION_OFFSET)

        rows_indicators.append(row_section_indicators)

    columns_indicators = [(columns_indicators[first_left], columns_indicators[first_right]),
                          (columns_indicators[second_left], columns_indicators[second_right])]
    rows_indicators = [rows_indicators[first_left], rows_indicators[second_left]]
    return columns_indicators, rows_indicators


def extract_lines(columns_indicators, rows_indicators, image_path, output_path):
    image_folder_path = os.path.splitext(image_path)[0]
    image_output_folder_path = os.path.join(output_path, os.path.basename(image_folder_path))

    try:
        os.mkdir(image_output_folder_path)
    except FileExistsError:
        shutil.rmtree(image_output_folder_path)
        os.mkdir(image_output_folder_path)

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    columns_name = ["left", "right"]
    clone = image.copy()

    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        cv2.line(clone, (column_left, 0), (column_left, height), (0, 255, 0), 1)
        cv2.line(clone, (column_right, 0), (column_right, height), (0, 255, 0), 1)

        for row_i in range(len(rows_indicators[column_index])):
            if row_i < len(rows_indicators[column_index]) - 1:
                x1 = columns_indicators[column_index][0]
                x2 = columns_indicators[column_index][1]
                y1 = rows_indicators[column_index][row_i]
                y2 = rows_indicators[column_index][row_i + 1]
                crop_img = clone[y1:y2, x1:x2]

                image_line_path = os.path.join(image_output_folder_path,
                                               "{}_line_{}.png".format(columns_name[column_index], row_i))
                cv2.imwrite(image_line_path, crop_img)

    cv2.imwrite(os.path.join(image_output_folder_path, os.path.basename(image_folder_path) + ".png"), image)


def process_dataset(dataset_name, greyscale=True):
    global max_label_length, number_equivalent

    training_data = []
    max_width = 0
    max_height = 0

    rearrange_numeration()

    dataset_path = "../dataset/lines/" + str(dataset_name) + "/0"

    with open("../dataset/" + str(dataset_name) + ".txt") as text_dataset:
        for line in text_dataset:
            line_string = line.replace("\n", "").lower()

            if line_string[0] == '_':
                if line_string[3] == '_':
                    page_number = line_string[2]
                    path = dataset_path + '0' + page_number
                    column_number = line_string[5]
                else:
                    page_number = line_string[2:4]
                    path = dataset_path + page_number
                    column_number = line_string[6]

                line_index = 0
                continue

            current_file = ("left_line_" if int(column_number) == 0 else "right_line_") + str(line_index) + ".png"
            img = cv2.imread(os.path.join(path, current_file))

            if greyscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if img.shape[0] > max_height:
                max_height = img.shape[0]
            if img.shape[1] > max_width:
                max_width = img.shape[1]

            line_index += 1
            numerical_sequence = []

            for char in line_string:
                if char not in LABELS_DICTIONARY.values():
                    LABELS_DICTIONARY[number_equivalent] = char
                    numerical_sequence.append(float(number_equivalent))
                    number_equivalent += 1
                else:
                    for key, val in LABELS_DICTIONARY.items():
                        if val == char:
                            numerical_sequence.append(float(key))

            training_data.append([img, numerical_sequence])

            if len(numerical_sequence) > max_label_length:
                max_label_length = len(numerical_sequence)

    text_dataset.close()

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)

    # naive normalization
    X = X / 255.0

    return X, y, max_width, max_height


def labels_decoding(sequence):
    return_string = ""

    for el in sequence:
        return_string += LABELS_DICTIONARY[el] + " "

    return return_string


def get_max_label_length():
    return max_label_length


def prepare_datasets(datasets, extract=False, greyscale=True):
    X = []
    y = []
    max_width = 0
    max_height = 0

    for dataset in datasets:
        if extract:
            input_path = pathlib.Path("../dataset/deskewed/" + dataset)
            output_path = pathlib.Path("../dataset/lines/" + dataset)

            if not input_path.exists():
                raise FileNotFoundError("Deskewed dataset not found. Please run `preprocessing.py` first")

            output_path.mkdir(parents=True, exist_ok=True)

            for file_name in sorted(os.listdir(input_path)):
                image_path = os.path.join(input_path, file_name)

                columns_indicators, rows_indicators = detect_lines(image_path)
                extract_lines(columns_indicators, rows_indicators, image_path, output_path)

        lines, labels, dataset_max_width, dataset_max_height = process_dataset(dataset, greyscale)

        X = np.append(X, lines)
        y.extend(labels)

        max_width = max(dataset_max_width, max_width)
        max_height = max(dataset_max_height, max_height)

    X = pad_data(X, max_width, max_height, 3)
    y = sequence.pad_sequences(y, value=float(0.0), dtype='float32', padding="post")

    save_pickle(X, "X_pad")
    save_pickle(y, "y_pad")
    save_json("dictionary.json", LABELS_DICTIONARY)

    return X, y
