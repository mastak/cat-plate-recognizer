import typing as t
from dataclasses import dataclass

import numpy as np


@dataclass
class Plate:
    image: np.ndarray
    characters: t.List[np.ndarray]
    coordinates: t.Tuple[int]
    number: str = None


class BaseHandler:
    def get_plates(self, img: np.ndarray, frame_index: int) -> t.List[Plate]:
        raise NotImplementedError
