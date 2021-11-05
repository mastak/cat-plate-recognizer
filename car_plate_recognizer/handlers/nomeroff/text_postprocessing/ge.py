import string

from .base import PostprocessManager


class Ge(PostprocessManager):
    name = "ge"

    ALLOWED_LITERS = [x for x in string.ascii_letters]
    ALLOWED_LITERS.append("0")

    STANDARDS = ["@@@###", "@@###@@"]
    STANDARD = ""

    def find(self, text: str, strong: bool = False) -> str:
        for standard in self.STANDARDS:
            self.STANDARD = standard
            match = self.find_fully(text)
            if match:
                text = match.group(0)
                newtext = ""
                for i, standart_letter in enumerate(standard):
                    if standart_letter[0] == "@" and text[i] == "0":
                        newtext += "O"
                    else:
                        newtext += text[i]
                return newtext
        return text
