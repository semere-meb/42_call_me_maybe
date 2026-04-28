import enum
import re
from typing import Callable, Pattern, TypedDict

import numpy as np
from colorama import Fore
from numpy.typing import NDArray

from src.errors import AppError
from src.model_wrapper import ModelWrapper


class States(enum.Enum):
    """

    Represents a state in the Finite Automata JSON schema, start to END or

    """

    START = "START"
    STR = "STR"
    STR_ESC = "STR_ESC"
    NBR_SIGN = "NBR_SIGN"
    NBR_PRE_DOT = "NBR_PRE_DOT"
    NBR_DOT = "NBR_DOT"
    NBR_POST_DOT = "NBR_POST_DOT"
    END = "END"
    ERROR = "ERROR"


patterns: dict[str, Pattern[str]] = {
    "space": re.compile(r"\s+"),
    "char": re.compile(r'[^"\\]+'),
    "digit": re.compile(r"\d+"),
    "negative_sign": re.compile(r"\s*-"),
    "literal": re.compile(r"\s*(true|false|null)\s*"),
    # structure controls
    "quote": re.compile(r'\s*"\s*'),
    "dot": re.compile(r"\."),
    "backslash": re.compile(r"\\"),
    "one_char": re.compile(r"."),
    # multiple alternation
    "nbr_terminator": re.compile(r"\s*\,?\s*}{,2}\s*"),
    "str_terminator": re.compile(r'"\s*\,?\s*}{,2}\s*'),
}


class StateTransition(TypedDict):
    """

    Defines a state, its valid token patterns and a function which routes the
    FA to the next state based on the token pattern.

    """

    valid_tokens: list[Pattern[str]]
    fn: Callable[[Pattern[str]], States]


_structural_patterns: set[Pattern[str]] = {
    patterns["str_terminator"],
    patterns["nbr_terminator"],
    patterns["quote"],
    patterns["space"],
}

transitions: dict[States, StateTransition] = {
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
            patterns["one_char"],
        ],
        "fn": lambda pattern: {
            patterns["one_char"]: States.STR,
        }.get(pattern, States.ERROR),
    },
}


class Schema:
    """

    The JSON schema representing the current state of the FA and a method to
    grab the value while it is in an acceptable state(s).

    """

    model: ModelWrapper
    transitions: dict[States, StateTransition]
    state: States

    def __init__(
        self,
        model: ModelWrapper,
        init_state: States = States.START,
        key_type: str = "unknown",
    ) -> None:
        self.model = model
        self.state = init_state
        self.token_count = 0

        if key_type == "string":
            self.transitions = {
                **transitions,
                States.START: {
                    "valid_tokens": [patterns["quote"]],
                    "fn": lambda pattern: {
                        patterns["quote"]: States.STR,
                    }.get(pattern, States.ERROR),
                },
            }
        else:
            self.transitions = transitions

    def get_next_val(self, string: str, max_token: int = 30) -> str:
        """

        returns a string (often multiple tokens) that take the FA to a
        acceptable terminating state. Starts next_valid_token with a top-k
        (k=20) tokens to focus on the top candidates.

        Args:
          string: str: The regressive request string the will eventually
              include the next val to grab yet another token
          max_token: int:  (Default value = 30) : upper bound to prevent
              infinite loop when a model goes astray.

        Returns:
          str: The next value for a parameter key,
              a string to that takes the FA to a valid END state.

        """

        input_ids = self.model.encode(string)

        res = ""
        while True:
            self.token_count += 1
            patterns: list[Pattern[str]] = self.transitions[self.state][
                "valid_tokens"
            ]
            fn: Callable[[Pattern[str]], States] = self.transitions[
                self.state
            ]["fn"]

            logits = np.array(
                self.model.get_logits(input_ids),
                dtype=float,
            )

            rank = logits.argsort()[::-1][:20]
            result = self.next_valid_token(rank, patterns, fn)
            if result is None:
                raise AppError("Can't find a valid token with top-k as 20")
            token_id, effective = result
            res += effective
            if (
                self.state in [States.END, States.ERROR]
                or self.token_count == max_token
            ):
                break
            input_ids.append(token_id)

        if self.state == States.ERROR:
            raise AppError("Invalid value")

        return res

    def next_valid_token(
        self,
        rank: NDArray[np.intp],
        patterns: list[Pattern[str]],
        transition_fn: Callable[[Pattern[str]], States],
    ) -> tuple[int, str] | None:
        """

        Selects the first token, from the top candidates top to down, that will
        bring the FA to a valid state.

        Args:
          rank: NDArray[np.intp]: The top 20 candidates.
          patterns: list[Pattern[str]]: The set of valid tokens for the
              current state.
          transition_fn: Callable[[Pattern[str]], States]: The routing function
              to transition to the next state.

        Returns: The token_id and the effective(valid part) of the decoded
            token.

        """
        for token_id in rank:
            if token_id not in self.model.vocab:
                continue
            token_text = self.model.vocab[token_id]["decoded"]
            print(
                f"\ttoken: {Fore.YELLOW}{repr(token_text):15.15}{Fore.RESET},"
                f"state: {Fore.YELLOW}{self.state:20.20}{Fore.RESET}",
                end="",
            )

            for split_pos in range(len(token_text), 0, -1):
                prefix = token_text[:split_pos]
                suffix = token_text[split_pos:]
                for pattern in patterns:
                    if not pattern.fullmatch(prefix):
                        continue
                    mid_state = transition_fn(pattern)
                    if not suffix:
                        self.state = mid_state
                        effective = (
                            "" if pattern in _structural_patterns else prefix
                        )
                        print(
                            f" ✔ split:{split_pos} new state:"
                            f" {Fore.CYAN}{self.state}{Fore.RESET}"
                        )
                        return int(token_id), effective
                    if mid_state not in self.transitions:
                        continue
                    next_patterns = self.transitions[mid_state]["valid_tokens"]
                    next_fn = self.transitions[mid_state]["fn"]
                    for next_pattern in next_patterns:
                        if next_pattern.fullmatch(suffix):
                            self.state = next_fn(next_pattern)
                            prefix_eff = (
                                ""
                                if pattern in _structural_patterns
                                else prefix
                            )
                            suffix_eff = (
                                ""
                                if next_pattern in _structural_patterns
                                else suffix
                            )
                            effective = prefix_eff + suffix_eff
                            print(
                                f" ✔ split:{split_pos} new state: "
                                f"{Fore.CYAN}{self.state}{Fore.RESET}"
                            )
                            return int(token_id), effective

            print(f"{Fore.RED} ✘ skipping{Fore.RESET}")
        return None
