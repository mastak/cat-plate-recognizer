"""
https://circuitdigest.com/microcontroller-projects/license-plate-recognition-using-raspberry-pi-and-opencv
"""

import logging
import typing as t

import imutils
import numpy as np
import pytesseract
from cv2 import cv2

from car_plate_recognizer.handlers.base import BaseHandler, Plate, save_img

logger = logging.getLogger(__name__)


class CircuitDigestHandler(BaseHandler):

    def get_plates(self, image: np.ndarray, frame_index: int) -> t.List[Plate]:
        plates = []
        image = cv2.resize(image, (620, 480))  # move to top level

        img_edged, img_gray = preprocess(image)
        contours = get_contours(img_edged)

        mask = np.zeros(img_gray.shape, np.uint8)
        for index, contour in enumerate(contours):
            new_image = cv2.drawContours(mask, [contour], 0, 255, -1)
            save_img(new_image, f'Circuit{frame_index}-{index}-new_image-1.jpg')
            new_image = cv2.bitwise_and(image, image, mask=mask)
            save_img(new_image, f'Circuit{frame_index}-{index}-new_image-2.jpg')

            # Now crop
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped = img_gray[topx:bottomx + 1, topy:bottomy + 1]

            # Read the number plate
            text = pytesseract.image_to_string(cropped, config='--psm 11')
            print("Detected Number is:", text)

            save_img(image, f'Circuit{frame_index}-{index}-origin.jpg')
            save_img(cropped, f'Circuit{frame_index}-{index}-cropped.jpg')

        return plates


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    return cv2.Canny(gray, 30, 200), gray  # Perform Edge detection


def get_contours(image):
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    filtered_contours = []

    # loop over our contours
    for contour in contours:
        # approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            filtered_contours.append(approx)

    # save_img(image, f'Circuit-before.jpg')
    for index, contour in enumerate(filtered_contours):
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
        # save_img(image, f'Circuit-after-{index}.jpg')
    # save_img(image, f'Circuit-after.jpg')

    # if detected == 1:
    #     cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    print("filtered_contours count", len(filtered_contours))
    return filtered_contours
