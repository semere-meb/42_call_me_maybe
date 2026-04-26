from __future__ import annotations

import sys

import pytest

from src.parser import parse_args


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = parse_args()
    assert args.output == "data/output/function_calling_results.json"
    assert args.input == "data/input/function_calling_tests.json"
    assert args.functions_definition == "data/input/functions_definition.json"
    assert args.model == "Qwen/Qwen3-0.6B"


def test_custom_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "-o", "custom/out.json"])
    assert parse_args().output == "custom/out.json"


def test_custom_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "-i", "my/input.json"])
    assert parse_args().input == "my/input.json"


def test_custom_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "-f", "my/defs.json"])
    assert parse_args().functions_definition == "my/defs.json"


def test_custom_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "-m", "mistral/7b"])
    assert parse_args().model == "mistral/7b"


def test_long_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--input", "a.json",
        "--output", "b.json",
        "--functions_definition", "c.json",
        "--model", "some/model",
    ])
    args = parse_args()
    assert args.input == "a.json"
    assert args.output == "b.json"
    assert args.functions_definition == "c.json"
    assert args.model == "some/model"
