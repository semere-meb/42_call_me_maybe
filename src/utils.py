import re

patterns = {
    "str": re.compile(r'^\s*[^"]*\s*$'),
    "nbr": re.compile(r"\d+"),
    "dot": re.compile(r"^\.$"),
    "sign": re.compile(r"^\s*[-,\+]\ws*$"),
    "literal": re.compile(r"^\s*true\s*$|^\s*false\s*$|^\s*null\s*$"),
    "curly_open": re.compile(r"^\s*{\s*$"),
    "curly_close": re.compile(r"^\s*}\s*$"),
    "square_open": re.compile(r"^\s*\[\s*$"),
    "square_close": re.compile(r"^\s*\]\s*$"),
    "colon": re.compile(r"^\s*:\s*$"),
    "comma": re.compile(r"^\s*,\s*$"),
    "space": re.compile(r"^\s+$"),
    "word": re.compile(r"^\s*[\w*'*']+\s*$"),
    "quote": re.compile(r'^\s*"\s*$'),
    # BPE
    "quote_colon": re.compile(r'^\s*" *: *$'),
    "quote_comma": re.compile(r'^\s*" *, *$'),
}
