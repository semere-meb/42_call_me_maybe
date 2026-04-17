import enum

from .utils import patterns


class States(enum.Enum):
    START = "start"
    OBJECT_OPEN = "object_open"
    KEY_OPENED = "key_opened"
    IN_KEY = "in_key"
    KEY_CLOSED = "key_closed"
    SEP_DONE = "sep_done"
    IN_OBJ_STR = "in_obj_str"
    VALUE_DONE = "value_done"
    ARRAY_OPEN = "array_open"
    PREV_VAL_DONE = "prev_val_done"
    IN_ARR_STR = "in_arr_str"
    OBJECT_CLOSED = "object_closed"
    VALID_STATE = "valid_state"
    OBJ_NBR_OPENED = "obj_nbr_opened"
    OBJ_NBR_DOT = "obj_nbr_dot"
    OBJ_NBR_CLOSED = "obj_nbr_closed"
    ARR_NBR_OPENED = "arr_nbr_opened"
    ARR_NBR_DOT = "arr_nbr_dot"
    ARR_NBR_CLOSED = "arr_nbr_closed"


class JSONSchema:
    def __init__(self) -> None:
        self.state = States.START
        self.stack = []

        self.transitions = {
            States.START: {
                "valid_tokens": [
                    patterns["curly_open"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["space"]: States.START,
                    patterns["curly_open"]: States.OBJECT_OPEN,
                }.get(pattern, "unknown"),
            },
            States.OBJECT_OPEN: {
                "valid_tokens": [
                    patterns["quote"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["quote"]: States.IN_KEY,
                    patterns["space"]: States.OBJECT_OPEN,
                }.get(pattern, "unknown"),
            },
            States.IN_KEY: {
                "valid_tokens": [
                    patterns["word"],
                    patterns["quote"],
                    patterns["quote_colon"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["word"]: States.IN_KEY,
                    patterns["quote"]: States.KEY_CLOSED,
                    patterns["quote_colon"]: States.SEP_DONE,
                    patterns["space"]: States.IN_KEY,
                }.get(pattern, "unknown"),
            },
            States.KEY_CLOSED: {
                "valid_tokens": [
                    patterns["colon"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["colon"]: States.SEP_DONE,
                    patterns["space"]: States.KEY_CLOSED,
                }.get(pattern, "unknown"),
            },
            States.SEP_DONE: {
                "valid_tokens": [
                    patterns["nbr"],
                    patterns["literal"],
                    patterns["square_open"],
                    patterns["curly_open"],
                    patterns["quote"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["nbr"]: States.OBJ_NBR_OPENED,
                    patterns["literal"]: States.VALUE_DONE,
                    patterns["square_open"]: States.ARRAY_OPEN,
                    patterns["curly_open"]: States.OBJECT_OPEN,
                    patterns["quote"]: States.IN_OBJ_STR,
                    patterns["space"]: States.SEP_DONE,
                }.get(pattern, "unknown"),
            },
            States.IN_OBJ_STR: {
                "valid_tokens": [
                    patterns["quote"],
                    patterns["quote_comma"],
                    patterns["str"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["quote"]: States.VALUE_DONE,
                    patterns["quote_comma"]: States.OBJECT_OPEN,
                    patterns["str"]: States.IN_OBJ_STR,
                    patterns["space"]: States.IN_OBJ_STR,
                }.get(pattern, "unknown"),
            },
            States.VALUE_DONE: {
                "valid_tokens": [
                    patterns["curly_close"],
                    patterns["comma"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["curly_close"]: States.OBJECT_CLOSED,
                    patterns["comma"]: States.OBJECT_OPEN,
                    patterns["space"]: States.VALUE_DONE,
                }.get(pattern, "unknown"),
            },
            States.OBJECT_CLOSED: {
                "valid_tokens": [
                    patterns["comma"],
                    patterns["curly_close"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["comma"]: States.START,
                    patterns["curly_close"]: States.OBJECT_CLOSED,
                    patterns["space"]: States.OBJECT_CLOSED,
                }.get(pattern, "unknown"),
            },
            States.ARRAY_OPEN: {
                "valid_tokens": [
                    patterns["quote"],
                    patterns["nbr"],
                    patterns["literal"],
                    patterns["square_close"],
                    patterns["curly_open"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["nbr"]: States.ARR_NBR_OPENED,
                    patterns["literal"]: States.PREV_VAL_DONE,
                    patterns["square_close"]: States.VALUE_DONE,
                    patterns["curly_open"]: States.OBJECT_OPEN,
                    patterns["quote"]: States.IN_ARR_STR,
                    patterns["space"]: States.ARRAY_OPEN,
                }.get(pattern, "unknown"),
            },
            States.PREV_VAL_DONE: {
                "valid_tokens": [
                    patterns["comma"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["comma"]: States.ARRAY_OPEN,
                    patterns["space"]: States.PREV_VAL_DONE,
                }.get(pattern, "unknown"),
            },
            States.IN_ARR_STR: {
                "valid_tokens": [
                    patterns["quote"],
                    patterns["space"],
                    patterns["str"],
                ],
                "fn": lambda pattern: {
                    patterns["quote"]: States.PREV_VAL_DONE,
                    patterns["space"]: States.IN_ARR_STR,
                    patterns["str"]: States.IN_ARR_STR,
                }.get(pattern, "unknown"),
            },
            States.ARR_NBR_OPENED: {
                "valid_tokens": [
                    patterns["nbr"],
                    patterns["dot"],
                    patterns["comma"],
                    patterns["square_close"],
                ],
                "fn": lambda pattern: {
                    patterns["nbr"]: States.ARR_NBR_OPENED,
                    patterns["dot"]: States.ARR_NBR_DOT,
                    patterns["comma"]: States.VALUE_DONE,
                    patterns["square_close"]: States.VALUE_DONE,
                }.get(pattern, "unknown"),
            },
            States.ARR_NBR_DOT: {
                "valid_tokens": [
                    patterns["nbr"],
                ],
                "fn": lambda pattern: {
                    patterns["nbr"]: States.ARR_NBR_CLOSED,
                }.get(pattern, "unknown"),
            },
            States.ARR_NBR_CLOSED: {
                "valid_tokens": [
                    patterns["nbr"],
                    patterns["square_close"],
                    patterns["comma"],
                ],
                "fn": lambda pattern: {
                    patterns["nbr"]: States.ARR_NBR_CLOSED,
                    patterns["square_close"]: States.VALUE_DONE,
                    patterns["comma"]: States.PREV_VAL_DONE,
                }.get(pattern, "unknown"),
            },
            States.OBJ_NBR_OPENED: {
                "valid_tokens": [
                    patterns["nbr"],
                    patterns["dot"],
                    patterns["comma"],
                    patterns["curly_close"],
                    patterns["space"],
                ],
                "fn": lambda pattern: {
                    patterns["nbr"]: States.OBJ_NBR_OPENED,
                    patterns["dot"]: States.OBJ_NBR_DOT,
                    patterns["comma"]: States.OBJECT_OPEN,
                    patterns["curly_close"]: States.OBJECT_CLOSED,
                    patterns["space"]: States.VALUE_DONE,
                }.get(pattern, "unknown"),
            },
            States.OBJ_NBR_DOT: {
                "valid_tokens": [
                    patterns["dot"],
                ],
                "fn": lambda pattern: {
                    patterns["dot"]: States.OBJ_NBR_CLOSED,
                }.get(pattern, "unknown"),
            },
            States.OBJ_NBR_CLOSED: {
                "valid_tokens": [
                    patterns["nbr"],
                    patterns["curly_close"],
                    patterns["comma"],
                ],
                "fn": lambda pattern: {
                    patterns["nbr"]: States.OBJ_NBR_CLOSED,
                    patterns["curly_close"]: States.OBJECT_CLOSED,
                    patterns["comma"]: States.OBJECT_OPEN,
                }.get(pattern, "unknown"),
            },
        }

    def ingest(self, id_rank, id_to_token):
        validators = self.transitions[self.state]["valid_tokens"]
        next_state_fn = self.transitions[self.state]["fn"]

        for token_id in id_rank:
            token_text = id_to_token[int(token_id)]
            for validator in validators:
                if validator.match(token_text):
                    next_state = next_state_fn(validator)
                    if patterns["curly_open"].match(token_text):
                        self.stack.append("{")
                    elif patterns["square_open"].match(token_text):
                        self.stack.append("[")
                    if patterns["curly_close"].match(token_text):
                        self.stack.pop()
                    elif patterns["square_close"].match(token_text):
                        self.stack.pop()
                    if self.state == States.OBJECT_CLOSED and not self.stack:
                        next_state = States.VALID_STATE

                    self.state = next_state
                    return token_id, token_text
