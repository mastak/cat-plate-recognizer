from .base import PostprocessManager


class EuUa1995(PostprocessManager):
    name = "eu_ua_1995"

    STANDARD = "#####@@"
    ALLOWED_LITERS = ["A", "B", "E", "I", "K", "M", "H", "O", "P", "C", "T", "X"]
    REPLACEMENT = {
        "#": {
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "Z": "2",  # 7
            "S": "5",  # 8
            "T": "7",
            "B": "8"
        },
        "@": {
            "/": "I",
            "|": "I",
            "L": "I",
            "1": "I",
            "5": "B",
            "8": "B",
            "R": "B",
            "0": "O",
            "Q": "O",
            "¥": "X",
            "Y": "X",
            "€": "C",
            "F": "E",
        },
    }

    def find(self, text: str, strong: bool = False) -> str:
        return super().find(text, strong)
