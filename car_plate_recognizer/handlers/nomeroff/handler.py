"""
based on https://github.com/mastak/nomeroff-net
"""
import logging
import typing as t

import numpy as np
from cv2 import cv2

from car_plate_recognizer.handlers.base import BaseHandler, Plate
from .bbox_np_points import NpPointsCraft
from .options_detector import OptionsDetector
from .text_detectors.eu import eu
from .text_postprocessing import postprocess_text
from .tools.image_processing import getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
from .yolov5_detector import Detector

logger = logging.getLogger(__name__)


OPTIONS_DETECTOR_CLASS_REGION = [
    "military",
    "eu_ua_2015",
    "eu_ua_2004",
    "eu_ua_1995",
    "eu",
    "xx_transit",
    "ru",
    "kz",
    "eu-ua-fake-dpr",
    "eu-ua-fake-lpr",
    "ge",
    "by",
    "su",
    "kg",
    "am",
]
OPTIONS_DETECTOR_COUNT_LINES = [1, 2, 3]


class NomeroffHandler(BaseHandler):
    def __init__(self):
        # self.detector = CustomDetector()
        self.detector = Detector()
        self.detector.load()

        self.npPointsCraft = NpPointsCraft()
        self.npPointsCraft.load()

        self.optionsDetector = OptionsDetector(
            options={
                "class_region": OPTIONS_DETECTOR_CLASS_REGION,
                "count_lines": OPTIONS_DETECTOR_COUNT_LINES,
            }
        )
        self.optionsDetector.load("latest")

        self.textDetector = eu()
        self.textDetector.load("latest")

    def get_plates(self, image: np.ndarray, frame_index: int) -> t.List[Plate]:
        # Detect numberplate
        # img_path = 'images/example2.jpeg'
        # img = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_boxes = self.detector.detect_bbox(image)
        # print("target_boxes", target_boxes)

        all_points = self.npPointsCraft.detect(image, target_boxes, [5, 2, 0])
        # print("all_points", all_points)

        # cut zones
        zones_rgb = [getCvZoneRGB(image, reshapePoints(rect, 1)) for rect in all_points]
        zones = convertCvZonesRGBtoBGR(zones_rgb)
        # print("all_points", zones)

        # predict zones attributes
        region_ids, count_lines = self.optionsDetector.predict(zones)
        # print("region_ids, count_lines", region_ids, count_lines)

        region_names = self.optionsDetector.getRegionLabels(region_ids)
        # print("region_names", region_names)

        # find text with postprocessing by standard
        text_arr = self.textDetector.predict(zones)
        # print("text_arr:", text_arr)
        text_arr = postprocess_text(text_arr, region_names)
        print("text_arr post:", text_arr)
        return []
