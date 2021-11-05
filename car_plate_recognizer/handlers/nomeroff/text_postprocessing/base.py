import logging
import re
import string

import numpy as np

logger = logging.getLogger(__name__)


class PostprocessManager:
    name = "xx_xx"
    processors = {}

    STANDARD = ""
    ALLOWED_LITERS = string.ascii_letters
    BLACK_LIST = ["\s", "\*", "\,", "\.", "\-", "'", '"', "\’", "_", "\+"]
    ALLOWED_NUMBERS = [str(item) for item in np.arange(10)]
    REPLACEMENT = {
        "#": {
            "I": "1",
            "Z": "2",  # 7
            "O": "0",
            "Q": "0",
            "B": "8",
            "D": "0",
            "S": "5",  # 8
            "T": "7"
        },
        "@": {
            "/": "I",
            "|": "I",
            "¥": "X",
            "€": "C"
        }
    }

    def __init_subclass__(cls, *args, **kwargs):
        if cls.name == "base":
            raise ValueError("Subclass has to set custom name")

        if cls.name in PostprocessManager.processors:
            raise ValueError(f"Processor with name {cls.name} already exists")

        if not re.match(r"^[#@]*$", cls.STANDARD):
            raise Exception(f"Standard {cls.STANDARD} not correct for {cls.name}")

        PostprocessManager.processors[cls.name] = cls()

    def process(self, key: str, text: str, strong: bool = True) -> str:
        if key not in PostprocessManager.processors:
            if key != self.name:
                logger.warning(f"Processor not found for key: {key}, using default processing")
            return self.find(text=text, strong=strong)
        return PostprocessManager.processors[key].find(text=text, strong=strong)

    def delete_all_black_list_characters(self, text: str) -> str:
        reg = "[{}]".format("".join(self.BLACK_LIST))
        return re.sub(re.compile(reg), "", text).replace("\\", "/").replace("\[", "|").replace("\]", "|")

    @staticmethod
    def check_pattern_standard(standard: str) -> str:
        if not re.match(r"^[#@]*$", standard):
            raise Exception("Standard {} not correct".format(standard))
        return standard

    @staticmethod
    def check_is_str(text: str) -> str:
        if type(text) is not str:
            raise ValueError("{} is not str".format(text))
        return text

    def find_fully(self, text: str):
        reg = ""
        for item in self.STANDARD:
            if item == "@":
                reg = "{}[{}]".format(reg, "".join(self.ALLOWED_LITERS))
            elif item == "#":
                reg = "{}[{}]".format(reg, "".join(self.ALLOWED_NUMBERS))
        reg_all = re.compile(reg)
        return re.search(reg_all, text)

    def replace(self, text: str) -> str:
        res = ""
        for i in np.arange(len(self.STANDARD)):
            l_dict = self.ALLOWED_LITERS
            if self.STANDARD[i] == "#":
                l_dict = self.ALLOWED_NUMBERS

            if text[i] in l_dict:
                res = "{}{}".format(res, text[i])
            else:
                replace_l = self.REPLACEMENT[self.STANDARD[i]][text[i]]
                res = "{}{}".format(res, replace_l)
        return res

    def find_similary(self, text: str) -> str:
        vcount = len(text) - len(self.STANDARD) + 1
        reg = ""
        for item in self.STANDARD:
            main = ""
            dop = ""
            if item == "@":
                dop = list(self.REPLACEMENT["@"].keys())
                main = self.ALLOWED_LITERS
            elif item == "#":
                dop = list(self.REPLACEMENT["#"].keys())
                main = self.ALLOWED_NUMBERS
            buf_reg = "".join(main + dop)
            reg = "{}[{}]".format(reg, buf_reg)
        reg_sim = re.compile(reg)
        for i in np.arange(vcount):
            buff_text = text[int(i) : int(len(self.STANDARD) + i)]
            match = re.search(reg_sim, buff_text)
            if match:
                return self.replace(match.group(0))
        return text

    def find(self, text: str, strong: bool = True) -> str:
        text = self.check_is_str(text)
        text = self.delete_all_black_list_characters(text)
        text = text.upper()

        if len(text) < len(self.STANDARD):
            return text

        if len(self.STANDARD):
            match = self.find_fully(text)
            if match:
                return match.group(0)

        if not strong:
            return self.find_similary(text)
        return text
