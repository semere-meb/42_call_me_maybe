import enum

from .utils import patterns


class States(enum.Enum):
    START = "start"
    OBJECT_OPEN = "object_open"
    KEY_OPENED = "key_opened"
    IN_KEY = "in_key"
    KEY_CLOSED = "key_closed"
    SEP_DONE = "sep_done"
    OBJ_STR_OPEN = "obj_str_open"
    IN_OBJ_STR = "in_obj_str"
    VALUE_DONE = "value_done"
    ARRAY_OPEN = "array_open"
    PREV_VAL_DONE = "prev_val_done"
    ARR_STR_OPEN = "arr_str_open"
    IN_ARR_STR = "in_arr_str"
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
                    patterns["space"],
                    patterns["curly_open"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.START,
                    patterns["curly_open"]: States.OBJECT_OPEN,
                }.get(pattern, "unknown"),
            },
            States.OBJECT_OPEN: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["quote"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.OBJECT_OPEN,
                    patterns["quote"]: States.IN_KEY,
                }.get(pattern, "unknown"),
            },
            States.IN_KEY: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["word"],
                    patterns["quote"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.IN_KEY,
                    patterns["word"]: States.IN_KEY,
                    patterns["quote"]: States.KEY_CLOSED,
                }.get(pattern, "unknown"),
            },
            States.KEY_CLOSED: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["colon"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.KEY_CLOSED,
                    patterns["colon"]: States.SEP_DONE,
                }.get(pattern, "unknown"),
            },
            States.SEP_DONE: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["nbr"],
                    patterns["null"],
                    patterns["bool"],
                    patterns["square_open"],
                    patterns["curly_open"],
                    patterns["quote"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.SEP_DONE,
                    patterns["nbr"]: States.VALUE_DONE,
                    patterns["null"]: States.VALUE_DONE,
                    patterns["bool"]: States.VALUE_DONE,
                    patterns["square_open"]: States.ARRAY_OPEN,
                    patterns["curly_open"]: States.OBJECT_OPEN,
                    patterns["quote"]: States.IN_OBJ_STR,
                }.get(pattern, "unknown"),
            },
            States.IN_OBJ_STR: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["word"],
                    patterns["quote"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.IN_OBJ_STR,
                    patterns["word"]: States.IN_OBJ_STR,
                    patterns["quote"]: States.VALUE_DONE,
                }.get(pattern, "unknown"),
            },
            States.VALUE_DONE: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["curly_close"],
                    patterns["comma"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.VALUE_DONE,
                    patterns["curly_close"]: States.VALID_JSON,
                    patterns["comma"]: States.OBJECT_OPEN,
                }.get(pattern, "unknown"),
            },
            States.ARRAY_OPEN: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["quote"],
                    patterns["nbr"],
                    patterns["null"],
                    patterns["bool"],
                    patterns["square_close"],
                    patterns["curly_open"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.ARRAY_OPEN,
                    patterns["nbr"]: States.PREV_VAL_DONE,
                    patterns["null"]: States.PREV_VAL_DONE,
                    patterns["bool"]: States.PREV_VAL_DONE,
                    patterns["square_close"]: States.VALUE_DONE,
                    patterns["curly_open"]: States.OBJECT_OPEN,
                    patterns["quote"]: States.IN_ARR_STR,
                }.get(pattern, "unknown"),
            },
            States.PREV_VAL_DONE: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["comma"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.PREV_VAL_DONE,
                    patterns["comma"]: States.ARRAY_OPEN,
                }.get(pattern, "unknown"),
            },
            States.IN_ARR_STR: {
                "valid_tokens": [
                    patterns["space"],
                    patterns["word"],
                    patterns["quote"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.IN_ARR_STR,
                    patterns["word"]: States.IN_ARR_STR,
                    patterns["quote"]: States.PREV_VAL_DONE,
                }.get(pattern, "unknown"),
            },
        }

    def ingest(self, candidates):
        validators = self.transitions[self.state]["valid_tokens"]
        next_state_fn = self.transitions[self.state]["fn"]

        for token_text, token_id, prob in candidates:
            for validator in validators:
                if validator.match(token_text):
                    # print(f"token: {repr(token_text)}, validator: {validator}")
                    self.state = next_state_fn(validator)
                    # TODO: handle updating the stack
                    # print(self.state)
                    return token_id
