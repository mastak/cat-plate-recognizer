"""
based on https://github.com/mastak/nomeroff-net
"""
import logging
import typing as t

import numpy as np
import pkg_resources
from cv2 import cv2

from NomeroffNet import textPostprocessing
from NomeroffNet.BBoxNpPoints import NpPointsCraft
from NomeroffNet.OptionsDetector import OptionsDetector
from NomeroffNet.TextDetectors.eu import eu
from NomeroffNet.YoloV5Detector import Detector
from NomeroffNet.tools.image_processing import getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
from car_plate_recognizer.handlers.base import BaseHandler, Plate

logger = logging.getLogger(__name__)

LATEST_MODEL_DETECTOR = "resources/Detector/yolov5/yolov5s-2021-07-28.pt"

LATEST_MODEL_NP_POINTS_CRAFT_MLT = "resources/NpPointsCraft/craft_mlt/craft_mlt_25k_2020-02-16.pth"
LATEST_MODEL_NP_POINTS_CRAFT_REFINDER = "resources/NpPointsCraft/craft_refiner/craft_refiner_CTW1500_2020-02-16.pth"

LATEST_MODEL_OPTIONS_DETECTOR = "resources/OptionsDetector/numberplate_options/numberplate_options_2021_08_13_pytorch_lightning.ckpt"

OPTIONS_DETECTOR_CLASS_REGION = ['military', 'eu_ua_2015', 'eu_ua_2004', 'eu_ua_1995', 'eu', 'xx_transit', 'ru', 'kz',
                                 'eu-ua-fake-dpr', 'eu-ua-fake-lpr', 'ge', 'by', 'su', 'kg', 'am']
OPTIONS_DETECTOR_COUNT_LINES = [1, 2, 3]


class CustomDetector(Detector):
    def load(self, path_to_model: str = "latest") -> None:
        if path_to_model == "latest":
            path_to_model = LATEST_MODEL_DETECTOR
        path_to_model = pkg_resources.resource_filename('car_plate_recognizer', path_to_model)
        return super().load(path_to_model)


class CustomNpPointsCraft(NpPointsCraft):
    def load(self, mtl_model_path: str = "latest", refiner_model_path: str = "latest") -> None:
        if mtl_model_path == "latest":
            mtl_model_path = LATEST_MODEL_NP_POINTS_CRAFT_MLT
        mtl_model_path = pkg_resources.resource_filename('car_plate_recognizer', mtl_model_path)

        if refiner_model_path == "latest":
            refiner_model_path = LATEST_MODEL_NP_POINTS_CRAFT_REFINDER
        refiner_model_path = pkg_resources.resource_filename('car_plate_recognizer', refiner_model_path)
        return super().load(mtl_model_path, refiner_model_path)


class CustomOptionsDetector(OptionsDetector):
    def load(self, path_to_model: str = "latest", options: t.Dict = None):
        if path_to_model == "latest":
            path_to_model = LATEST_MODEL_OPTIONS_DETECTOR
        path_to_model = pkg_resources.resource_filename('car_plate_recognizer', path_to_model)
        return super().load(path_to_model, options)


class NomeroffHandler(BaseHandler):
    def __init__(self):
        self.detector = CustomDetector()
        self.detector.load()

        self.npPointsCraft = CustomNpPointsCraft()
        self.npPointsCraft.load()

        self.optionsDetector = CustomOptionsDetector(options={
            'class_region': OPTIONS_DETECTOR_CLASS_REGION,
            'count_lines': OPTIONS_DETECTOR_COUNT_LINES,
        })
        self.optionsDetector.load("latest")

        self.textDetector = eu()
        self.textDetector.load("latest")

    def get_plates(self, image: np.ndarray, frame_index: int) -> t.List[Plate]:
        # Detect numberplate
        # img_path = 'images/example2.jpeg'
        # img = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_boxes = self.detector.detect_bbox(image)
        print("target_boxes", target_boxes)

        all_points = self.npPointsCraft.detect(image, target_boxes, [5, 2, 0])
        print("all_points", all_points)

        # cut zones
        zones_rgb = [getCvZoneRGB(image, reshapePoints(rect, 1)) for rect in all_points]
        zones = convertCvZonesRGBtoBGR(zones_rgb)
        # print("all_points", zones)

        # predict zones attributes
        region_ids, count_lines = self.optionsDetector.predict(zones)
        print("region_ids, count_lines", region_ids, count_lines)

        region_names = self.optionsDetector.getRegionLabels(region_ids)
        print("region_names", region_names)

        # find text with postprocessing by standard
        text_arr = self.textDetector.predict(zones)
        print("text_arr:", text_arr)
        text_arr = textPostprocessing(text_arr, region_names)
        print("text_arr post:", text_arr)
        return []
