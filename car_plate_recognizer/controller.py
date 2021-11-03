import logging
import time

import coloredlogs
import numpy as np
# from PIL import Image
from cv2 import cv2

from car_plate_recognizer.handlers.neural_1.handler import Neural1Handler

logger = logging.getLogger(__name__)

coloredlogs.install(level=logging.INFO)


def get_time() -> float:
    """ Return time in ms"""
    return time.perf_counter() * 1000


def stream_handler():
    # Initialize the Neural Network
    handlers = (
        Neural1Handler(),
    )

    frame_index = 0
    cap = cv2.VideoCapture("/Users/ihorliubymov/Downloads/video.MOV")
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image: np.ndarray = image
        frame_index += 1

        # Image.fromarray(image).save(f"logs/{frame_index}-frame-src.jpg")
        for handler in handlers:
            started_at = get_time()
            plates = handler.get_plates(image, frame_index)
            spent_time = get_time() - started_at

            if not plates:
                logger.info(f'{handler.__class__.__name__} took {spent_time}, car number not found')

            for index, plate in enumerate(plates):
                logger.info(f'{handler.__class__.__name__} took {spent_time}, found a car number: {plate.number}')
                # Image.fromarray(plate.image).save(f"logs/{frame_index}-{index}-plate.jpg")

    cap.release()
    cv2.destroyAllWindows()
