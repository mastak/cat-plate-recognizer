from typing import List

from . import eu_ua_1995
from . import eu_ua_2004
from . import eu_ua_2004_squire
from . import eu_ua_2015
from . import eu_ua_2015_squire
from . import ge
from .base import PostprocessManager

process_manager = PostprocessManager()

CYRILLIC_MAP = {
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "І": "I",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Т": "T",
    "Х": "X",
}


def translate_cyrillic_to_latin(cyrillic_str):
    cyrillic_str = cyrillic_str.upper()
    latin_str = ""
    for litter in cyrillic_str:
        latin_str = f"{latin_str}{CYRILLIC_MAP.get(litter, litter)}"
    return latin_str


def postprocess_text(texts: List[str], text_postprocess_names: List[str]) -> List[str]:
    res_texts = []
    for text, textPostprocessName in zip(texts, text_postprocess_names):
        text = translate_cyrillic_to_latin(text)

        _textPostprocessName = textPostprocessName.replace("-", "_")
        text = process_manager.process(key=_textPostprocessName, text=text)

        res_texts.append(text)
    return res_texts
