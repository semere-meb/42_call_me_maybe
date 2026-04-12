import re
import enum

from .utils import patterns


class States(enum.Enum):
    START = "start"
    OBJECT_OPEN = "object_open"
    KEY_DONE = "key_done"
    SEP_DONE = "sep_done"
    VALUE_DONE = "value_done"
    ARRAY_OPEN = "array_open"
    PREV_VAL_DONE = "prev_val_done"
    VALID_JSON = "valid_json"


class JSONSchema:
    def __init__(self) -> None:
        self.state = States.START
        self.output = []
        self.stack = []
        self.stack_depth = 0

        self.transitions = {
            States.START: {
                "valid_tokens": [
                    patterns["curly_open"],
                ],
                "fn": lambda pattern: {
                    patterns["curly_open"]: States.OBJECT_OPEN,
                }.get(pattern, "unknown"),
            },
            States.OBJECT_OPEN: {
                "valid_tokens": [
                    patterns["str"],
                ],
                "fn": lambda pattern: {
                    patterns["str"]: States.KEY_DONE,
                }.get(pattern, "unknown"),
            },
            States.KEY_DONE: {
                "valid_tokens": [
                    patterns["colon"],
                ],
                "fn": lambda pattern: {
                    patterns["colon"]: States.SEP_DONE,
                }.get(pattern, "unknown"),
            },
            States.SEP_DONE: {
                "valid_tokens": [
                    patterns["str"],
                    patterns["number"],
                    patterns["null"],
                    patterns["bool"],
                    patterns["square_open"],
                    patterns["curly_open"],
                ],
                "fn": lambda pattern: {
                    patterns["str"]: States.VALUE_DONE,
                    patterns["number"]: States.VALUE_DONE,
                    patterns["null"]: States.VALUE_DONE,
                    patterns["bool"]: States.VALUE_DONE,
                    patterns["square_open"]: States.ARRAY_OPEN,
                    patterns["curly_open"]: States.OBJECT_OPEN,
                }.get(pattern, "unknown"),
            },
            States.VALUE_DONE: {
                "valid_tokens": [
                    patterns["curly_close"],
                ],
                "fn": lambda pattern: {
                    patterns["curly_close"]: States.VALID_JSON,
                }.get(pattern, "unknown"),
            },
            States.ARRAY_OPEN: {
                "valid_tokens": [
                    patterns["str"],
                    patterns["number"],
                    patterns["null"],
                    patterns["bool"],
                    patterns["square_close"],
                    patterns["curly_open"],
                    patterns["comma"],
                ],
                "fn": lambda pattern: {
                    patterns["str"]: States.PREV_VAL_DONE,
                    patterns["number"]: States.PREV_VAL_DONE,
                    patterns["null"]: States.PREV_VAL_DONE,
                    patterns["bool"]: States.PREV_VAL_DONE,
                    patterns["square_close"]: States.VALUE_DONE,
                    patterns["curly_open"]: States.OBJECT_OPEN,
                    patterns["comma"]: States.ARRAY_OPEN,
                }.get(pattern, "unknown"),
            },
        }

    def is_valid(self, token):
        return any(
            pattern.match(token) for pattern in self.transitions[self.state]
        )

    def get_next_state(self, token):
        return self.transitions[self.state]["fn"](token)

    def update_stack(self, token):
        if patterns["curly_open"].match(token) or patterns[
            "square_close"
        ].match(token):
            self.stack += 1
        elif patterns["curly_close"].match(token) or patterns[
            "sqaure_close"
        ].match(token):
            self.stack -= 1
