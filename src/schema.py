import enum
import re
from typing import AnyStr, Callable, Pattern

import numpy as np

from llm_sdk import Small_LLM_Model
from src.errors import AppError


class States(enum.Enum):
    START = "START"
    STR = "STR"
    STR_ESC = "STR_ESC"
    NBR_SIGN = "NBR_SIGN"
    NBR_PRE_DOT = "NBR_PRE_DOT"
    NBR_DOT = "NBR_DOT"
    NBR_POST_DOT = "NBR_POST_DOT"
    END = "END"
    ERROR = "ERROR"


patterns = {
    "space": re.compile(r"\s+"),
    "char": re.compile(r'[^"\\]*'),
    "digit": re.compile(r"\d+"),
    "negative_sign": re.compile(r"\s*-"),
    "literal": re.compile(r"\s*(true|false|null)\s*"),
    # structure controls
    "quote": re.compile(r'\s*"\s*'),
    "comma": re.compile(r"\s*,\s*$"),
    "curly_close": re.compile(r"\s*}+\s*"),
    "dot": re.compile(r"\."),
    "backslash": re.compile(r"\\"),
    "one_char": re.compile(r"."),
    # multiple alternation
    "nbr_terminator": re.compile(r"\s*\,?\s*}{,2}\s*"),
    "str_terminator": re.compile(r'"\s*\,?\s*}{,2}\s*'),
}

transitions: dict[
    States, dict[str, list | Callable[[Pattern[AnyStr]], States]]
] = {
    States.START: {
        "valid_tokens": [
            patterns["digit"],
            patterns["quote"],
            patterns["literal"],
            patterns["negative_sign"],
            patterns["space"],
        ],
        "fn": lambda pattern: {
            patterns["quote"]: States.STR,
            patterns["literal"]: States.END,
            patterns["negative_sign"]: States.NBR_SIGN,
            patterns["digit"]: States.NBR_PRE_DOT,
            patterns["space"]: States.START,
        }.get(pattern, States.ERROR),
    },
    States.NBR_SIGN: {
        "valid_tokens": [
            patterns["digit"],
        ],
        "fn": lambda pattern: {
            patterns["digit"]: States.NBR_PRE_DOT,
        }.get(pattern, States.ERROR),
    },
    States.NBR_PRE_DOT: {
        "valid_tokens": [
            patterns["digit"],
            patterns["dot"],
            patterns["nbr_terminator"],
        ],
        "fn": lambda pattern: {
            patterns["digit"]: States.NBR_PRE_DOT,
            patterns["dot"]: States.NBR_DOT,
            patterns["nbr_terminator"]: States.END,
        }.get(pattern, States.ERROR),
    },
    States.NBR_DOT: {
        "valid_tokens": [
            patterns["digit"],
        ],
        "fn": lambda pattern: {
            patterns["digit"]: States.NBR_POST_DOT,
        }.get(pattern, States.ERROR),
    },
    States.NBR_POST_DOT: {
        "valid_tokens": [
            patterns["digit"],
            patterns["nbr_terminator"],
        ],
        "fn": lambda pattern: {
            patterns["digit"]: States.NBR_POST_DOT,
            patterns["nbr_terminator"]: States.END,
        }.get(pattern, States.ERROR),
    },
    States.STR: {
        "valid_tokens": [
            patterns["backslash"],
            patterns["str_terminator"],
            patterns["char"],
        ],
        "fn": lambda pattern: {
            patterns["backslash"]: States.STR_ESC,
            patterns["str_terminator"]: States.END,
            patterns["char"]: States.STR,
        }.get(pattern, States.ERROR),
    },
    States.STR_ESC: {
        "valid_tokens": [
            patterns["char"],
        ],
        "fn": lambda pattern: {
            patterns["char"]: States.STR,
        }.get(pattern, States.ERROR),
    },
}


class Schema:
    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: dict[int, dict[str, str]],
        init_state: States = States.START,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.state = init_state
        self.token_count = 0
        self.transitions: dict[
            States, dict[str, list | Callable[[Pattern], States]]
        ] = transitions

    def get_next_val(self, string, max_token: int = -1) -> str:

        input_ids = self.model.encode(string)[0].tolist()

        res = ""
        while (
            self.state not in [States.END, States.ERROR]
            or self.token_count == max_token
        ):
            validators: list[Pattern[AnyStr]] = self.transitions[self.state][
                "valid_tokens"
            ]
            transition_fn: Callable[[Pattern[AnyStr]], States] = (
                self.transitions[self.state]["fn"]
            )

            logits = np.array(
                self.model.get_logits_from_input_ids(input_ids),
                dtype=float,
            )

            # top-k, with k = 20
            rank = logits.argsort()[::-1][:20]
            token_id = self.next_valid_token(rank, validators, transition_fn)
            if token_id is None:
                raise AppError("Can't find a valid token with top-k as 20")
            if self.state in [States.END, States.ERROR]:
                break
            input_ids.append(token_id)
            res += self.vocab[token_id]["decoded"]

        if self.state == States.ERROR:
            raise AppError("Invalid value")

        return res

    def next_valid_token(self, rank, validators, transition_fn) -> int | None:
        for token_id in rank:
            token_text = self.vocab[token_id]["decoded"]
            print(f"Trying {repr(token_text)}, {self.state}")
            for validator in validators:
                if validator.fullmatch(token_text):
                    print("MATCHED")
                    self.state = transition_fn(validator)
                    return token_id
        return None
