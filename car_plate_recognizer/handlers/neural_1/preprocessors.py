import typing as t

import imutils
import numpy as np
from cv2 import cv2
from skimage import measure
from skimage.filters import threshold_local

from car_plate_recognizer.handlers.base import Plate

ELEMENT_STRUCTURE = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

MIN_AREA = 4500  # minimum area of the plate
MAX_AREA = 30000  # maximum area of the plate


def find_possible_plates(input_img) -> t.Optional[t.List[Plate]]:
    """
    Finding all possible contours that can be plates
    """
    plates = []
    img_preprocessed = preprocess_color(input_img)
    possible_plate_contours = get_contours(img_preprocessed)

    for plate_contour in possible_plate_contours:
        plate = check_plate_contour(input_img, plate_contour)
        if plate is not None:
            plates.append(plate)

    return plates


def preprocess_color(img):
    img_blurred = cv2.GaussianBlur(img, (7, 7), 0)  # old window was (5,5)
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)  # convert to gray
    # sobelX to get the vertical edges
    sobelx = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3)
    ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    morph_n_threshold_img = threshold_img.copy()
    cv2.morphologyEx(
        src=threshold_img,
        op=cv2.MORPH_CLOSE,
        kernel=ELEMENT_STRUCTURE,
        dst=morph_n_threshold_img,
    )
    return morph_n_threshold_img


def clean_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours = get_contours(thresh.copy())

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        # index of the largest contour in the area array
        max_index = np.argmax(areas)

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)
        # rect = cv2.minAreaRect(max_cnt)
        if not is_valid_ratio(max_cntArea, plate.shape[1], plate.shape[0]):
            return plate, False, None
        return plate, True, [x, y, w, h]
    else:
        return plate, False, None


def check_plate_contour(input_img, contour):
    min_rect = cv2.minAreaRect(contour)
    if not is_valid_rect(min_rect):
        return None

    x, y, w, h = cv2.boundingRect(contour)
    after_validation_img = input_img[y: y + h, x: x + w]
    after_clean_plate_img, is_plate_found, coordinates = clean_plate(after_validation_img)
    if not is_plate_found:
        return None

    characters_on_plate = segment_chars(after_clean_plate_img, 400)
    if characters_on_plate is None or len(characters_on_plate) != 8:  # TODO: len(characters_on_plate) != 8???
        return None

    x1, y1, w1, h1 = coordinates
    coordinates = x1 + x, y1 + y
    return Plate(
        image=after_clean_plate_img,
        characters=characters_on_plate,
        coordinates=coordinates,
    )


def sort_cont(character_contours):
    """
    To sort contours from left to right
    """
    i = 0
    bounding_boxes = [cv2.boundingRect(c) for c in character_contours]
    character_contours, _ = zip(
        *sorted(
            zip(character_contours, bounding_boxes),
            key=lambda b: b[1][i],
            reverse=False,
        )
    )
    return character_contours


def segment_chars(plate_img, fixed_width):
    """
    extract Value channel from the HSV format of image and apply adaptive thresholding
    to reveal the characters on the license plate
    """
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    T = threshold_local(V, 29, offset=15, method="gaussian")

    thresh = (V > T).astype("uint8") * 255

    thresh = cv2.bitwise_not(thresh)

    # resize the license plate region to a canoncial size
    # TODO: if plate_img.size != fixed_width:
    plate_img = imutils.resize(plate_img, width=fixed_width)
    # TODO: if thresh.size != fixed_width:
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # perform a connected components analysis and initialize the mask to store the locations
    # of the character candidates
    labels = measure.label(thresh, background=0)

    char_candidates = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    characters = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255

        cnts = get_contours(labelMask, method=cv2.CHAIN_APPROX_SIMPLE)

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:

            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solodity, and height ration for the component
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)

                cv2.drawContours(char_candidates, [hull], -1, 255, -1)

    contours = get_contours(char_candidates, method=cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4  # value to be added to each dimension of the character
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if y > addPixel:
                y = y - addPixel
            else:
                y = 0
            if x > addPixel:
                x = x - addPixel
            else:
                x = 0
            temp = bgr_thresh[y: y + h + (addPixel * 2), x: x + w + (addPixel * 2)]

            characters.append(temp)
        return characters
    else:
        return None


def is_valid_ratio(
        area,
        width,
        height,
        ratio_min: t.Union[int, float] = 3,
        ratio_max: t.Union[int, float] = 6,
):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    if (area < MIN_AREA or area > MAX_AREA) or (ratio < ratio_min or ratio > ratio_max):
        return False
    return True


def is_valid_rect(rect):
    (x, y), (width, height), rect_angle = rect

    if width > height:
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 15:
        return False
    if height == 0 or width == 0:
        return False

    area = width * height
    if not is_valid_ratio(area, width, height, ratio_min=2.5, ratio_max=7):
        return False
    else:
        return True


def get_contours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE):
    contours, _ = cv2.findContours(image=image, mode=mode, method=method)
    return contours
