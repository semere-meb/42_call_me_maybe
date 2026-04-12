import re

patterns = {
    "str_pattern": re.compile(r'^\s*".*"\s*$'),
    "nbr_pattern": re.compile(r"^\s*\w+(\.\w+)?\s*$"),
    "bool_pattern": re.compile(r"^\s*true\s*$|^\s*false\s*$"),
    "null_pattern": re.compile(r"^\s*null\s*$"),
    "curly_open_pattern": re.compile(r"^\s*{\s*$"),
    "curly_close_pattern": re.compile(r"^\s*}\s*$"),
    "square_open_pattern": re.compile(r"^\s*[\s*$"),
    "square_close_pattern": re.compile(r"^\s*]\s*$"),
    "colon_pattern": re.compile(r"^\s*:\s*$"),
    "comma_pattern": re.compile(r"^\s*,\s*$"),
    "space_pattern": re.compile(r"^\s+$"),
}
