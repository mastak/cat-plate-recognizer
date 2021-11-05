import sys
from pathlib import Path

from . import tools

ROOT_PATH = Path(__file__).parent.parent.parent.parent

sys.path.append(ROOT_PATH.joinpath("libs").as_posix())
sys.path.append(ROOT_PATH.joinpath("libs").joinpath("yolov5").as_posix())

sys.modules['NomeroffNet.tools'] = tools  # added link for correct unpickling
sys.modules['NomeroffNet'] = 'FAKE'  # fake link for correct unpickling
