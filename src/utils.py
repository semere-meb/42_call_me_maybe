import re

patterns = {
    "str": re.compile(r'^\s*".*"\s*$'),
    "nbr": re.compile(r"^\s*\w+(\.\w+)?\s*$"),
    "bool": re.compile(r"^\s*true\s*$|^\s*false\s*$"),
    "null": re.compile(r"^\s*null\s*$"),
    "curly_open": re.compile(r"^\s*{\s*$"),
    "curly_close": re.compile(r"^\s*}\s*$"),
    "square_open_pattern": re.compile(r"^\s*[\s*$"),
    "square_close": re.compile(r"^\s*]\s*$"),
    "colon": re.compile(r"^\s*:\s*$"),
    "comma": re.compile(r"^\s*,\s*$"),
    "space": re.compile(r"^\s+$"),
}
