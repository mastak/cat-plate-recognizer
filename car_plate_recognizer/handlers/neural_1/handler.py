import typing as t
import logging

import numpy as np


from car_plate_recognizer.handlers.base import BaseHandler, Plate
from .preprocessors import preprocess_color, get_contours, check_plate_contour
from .ml_model import NeuralNetwork

logger = logging.getLogger(__name__)


class Neural1Handler(BaseHandler):
    def __init__(self):
        self.model = NeuralNetwork()

    def get_plates(self, image: np.ndarray, frame_index: int) -> t.List[Plate]:
        plates = []
        possible_plates = find_possible_plates(image)
        for index, plate in enumerate(possible_plates):
            number = self.model.label_image_list(plate.characters, imageSizeOuput=128)
            if number:
                plate.number = number
                plates.append(plate)
        return plates


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
