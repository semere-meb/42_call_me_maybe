from __future__ import annotations

import pytest

from src.errors import AppError
from src.schema import Schema, States


class _MockModel:
    """Minimal ModelWrapper stand-in for Schema tests — no LLM required."""

    def __init__(
        self,
        vocab: dict[int, dict[str, str]],
        logit_rounds: list[dict[int, float]],
    ) -> None:
        self.vocab = vocab
        self._rounds = iter(logit_rounds)

    def encode(self, text: str) -> list[int]:
        return [0]

    def get_logits(self, input_ids: list[int]) -> list[float]:
        scores = next(self._rounds)
        size = max(self.vocab) + 1
        logits: list[float] = [-1e9] * size
        for tid, score in scores.items():
            logits[tid] = score
        return logits

    def decode(self, token_id: int) -> str:
        return self.vocab[token_id]["decoded"]


# ── strings ──────────────────────────────────────────────────────────────────


def test_string_single_token() -> None:
    vocab = {
        1: {"raw": "hello", "decoded": "hello"},
        2: {"raw": '",', "decoded": '",'},
    }
    schema = Schema(_MockModel(vocab, [{1: 1.0}, {2: 1.0}]), States.STR)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "hello"


def test_string_multi_token() -> None:
    vocab = {
        1: {"raw": "foo", "decoded": "foo"},
        2: {"raw": "bar", "decoded": "bar"},
        3: {"raw": '"', "decoded": '"'},
    }
    schema = Schema(_MockModel(vocab, [{1: 1.0}, {2: 1.0}, {3: 1.0}]), States.STR)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "foobar"


def test_string_with_spaces() -> None:
    vocab = {
        1: {"raw": "hello world", "decoded": "hello world"},
        2: {"raw": '"', "decoded": '"'},
    }
    schema = Schema(_MockModel(vocab, [{1: 1.0}, {2: 1.0}]), States.STR)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "hello world"


def test_string_started_from_start_state() -> None:
    """Opening quote transitions START → STR; content follows."""
    vocab = {
        1: {"raw": '"', "decoded": '"'},
        2: {"raw": "hi", "decoded": "hi"},
        3: {"raw": '"}', "decoded": '"}'},
    }
    schema = Schema(_MockModel(vocab, [{1: 1.0}, {2: 1.0}, {3: 1.0}]), States.START)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "hi"


# ── split tokens ─────────────────────────────────────────────────────────────


def test_split_content_then_terminator() -> None:
    """Token ')"': content ')' fused with terminator '"'."""
    vocab = {1: {"raw": ')"', "decoded": ')"'}}
    schema = Schema(_MockModel(vocab, [{1: 1.0}]), States.STR)  # type: ignore[arg-type]
    assert schema.get_next_val("") == ")"


def test_split_structural_then_content() -> None:
    """Token ' \"/': opening quote (structural) fused with content '/'."""
    vocab = {
        1: {"raw": ' "/', "decoded": ' "/'},
        2: {"raw": '",', "decoded": '",'},
    }
    schema = Schema(_MockModel(vocab, [{1: 1.0}, {2: 1.0}]), States.START)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "/"


def test_split_content_then_backslash() -> None:
    """Token ':\\': content ':' fused with backslash that starts an escape."""
    vocab = {
        1: {"raw": ":\\", "decoded": ":\\"},
        2: {"raw": "U", "decoded": "U"},
        3: {"raw": '",', "decoded": '",'},
    }
    schema = Schema(_MockModel(vocab, [{1: 1.0}, {2: 1.0}, {3: 1.0}]), States.STR)  # type: ignore[arg-type]
    assert schema.get_next_val("") == ":\\U"


def test_skips_invalid_then_matches_next() -> None:
    """Top-ranked token is invalid; second-ranked is accepted."""
    vocab = {
        1: {"raw": "x", "decoded": "x"},
        2: {"raw": "5", "decoded": "5"},
        3: {"raw": "}", "decoded": "}"},
    }
    # token 1 (score 2.0) is top-ranked but 'x' is not a digit in NBR_DOT
    schema = Schema(  # type: ignore[arg-type]
        _MockModel(vocab, [{1: 2.0, 2: 1.0}, {3: 1.0}]),
        States.NBR_DOT,
    )
    assert schema.get_next_val("") == "5"


# ── numbers ──────────────────────────────────────────────────────────────────


def test_number_integer() -> None:
    vocab = {
        1: {"raw": "42", "decoded": "42"},
        2: {"raw": "}", "decoded": "}"},
    }
    schema = Schema(_MockModel(vocab, [{1: 1.0}, {2: 1.0}]), States.START)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "42"


def test_number_float() -> None:
    vocab = {
        1: {"raw": "3", "decoded": "3"},
        2: {"raw": ".", "decoded": "."},
        3: {"raw": "14", "decoded": "14"},
        4: {"raw": "}", "decoded": "}"},
    }
    schema = Schema(  # type: ignore[arg-type]
        _MockModel(vocab, [{1: 1.0}, {2: 1.0}, {3: 1.0}, {4: 1.0}]),
        States.START,
    )
    assert schema.get_next_val("") == "3.14"


def test_number_negative() -> None:
    vocab = {
        1: {"raw": "-", "decoded": "-"},
        2: {"raw": "7", "decoded": "7"},
        3: {"raw": "}", "decoded": "}"},
    }
    schema = Schema(  # type: ignore[arg-type]
        _MockModel(vocab, [{1: 1.0}, {2: 1.0}, {3: 1.0}]),
        States.START,
    )
    assert schema.get_next_val("") == "-7"


def test_number_comma_terminator() -> None:
    """nbr_terminator also matches ',}' between parameters."""
    vocab = {
        1: {"raw": "99", "decoded": "99"},
        2: {"raw": ",}", "decoded": ",}"},
    }
    schema = Schema(_MockModel(vocab, [{1: 1.0}, {2: 1.0}]), States.START)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "99"


# ── literals ─────────────────────────────────────────────────────────────────


def test_literal_true() -> None:
    vocab = {1: {"raw": "true", "decoded": "true"}}
    schema = Schema(_MockModel(vocab, [{1: 1.0}]), States.START)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "true"


def test_literal_false() -> None:
    vocab = {1: {"raw": "false", "decoded": "false"}}
    schema = Schema(_MockModel(vocab, [{1: 1.0}]), States.START)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "false"


def test_literal_null() -> None:
    vocab = {1: {"raw": "null", "decoded": "null"}}
    schema = Schema(_MockModel(vocab, [{1: 1.0}]), States.START)  # type: ignore[arg-type]
    assert schema.get_next_val("") == "null"


# ── error / edge cases ───────────────────────────────────────────────────────


def test_no_valid_token_raises() -> None:
    # NBR_DOT only accepts digits; all tokens decode to 'x'
    vocab = {i: {"raw": "x", "decoded": "x"} for i in range(1, 22)}
    schema = Schema(  # type: ignore[arg-type]
        _MockModel(vocab, [{i: float(i) for i in range(1, 22)}]),
        States.NBR_DOT,
    )
    with pytest.raises(AppError, match="Can't find a valid token"):
        schema.get_next_val("")


def test_max_token_stops_generation() -> None:
    vocab = {1: {"raw": "a", "decoded": "a"}}
    schema = Schema(_MockModel(vocab, [{1: 1.0}] * 10), States.STR)  # type: ignore[arg-type]
    assert schema.get_next_val("", max_token=3) == "aaa"
