import os
import typing as t
from dataclasses import dataclass

import numpy as np
from PIL import Image

BASE_PATH = "/Users/ihorliubymov/source/mastak/car-plate-recognizer/logs"


@dataclass
class Plate:
    image: np.ndarray
    characters: t.List[np.ndarray]
    coordinates: t.Tuple[int]
    number: str = None


class BaseHandler:
    def get_plates(self, img: np.ndarray, frame_index: int) -> t.List[Plate]:
        raise NotImplementedError


def save_img(image: np.ndarray, file_name):
    Image.fromarray(image).save(os.path.join(BASE_PATH, file_name))
