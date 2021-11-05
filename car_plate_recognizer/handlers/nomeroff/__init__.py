import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent.parent

print("ROOT_PATH", ROOT_PATH)
sys.path.append(ROOT_PATH.joinpath("libs").as_posix())
sys.path.append(ROOT_PATH.joinpath("libs").joinpath("yolov5").as_posix())
