from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from src.errors import AppError
from src.io_utils import flush_results, get_definitions, get_files, get_prompts


# ── get_prompts ───────────────────────────────────────────────────────────────


def test_get_prompts_valid(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps([{"prompt": "What is 2+2?"}]))
    prompts = get_prompts(f)
    assert len(prompts) == 1
    assert prompts[0].prompt == "What is 2+2?"


def test_get_prompts_multiple(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps([{"prompt": "A"}, {"prompt": "B"}]))
    prompts = get_prompts(f)
    assert [p.prompt for p in prompts] == ["A", "B"]


def test_get_prompts_skips_empty_dicts(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps([{}, {"prompt": "hello"}]))
    prompts = get_prompts(f)
    assert len(prompts) == 1
    assert prompts[0].prompt == "hello"


def test_get_prompts_empty_array_raises(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text("[]")
    with pytest.raises(AppError, match="No inputs"):
        get_prompts(f)


def test_get_prompts_invalid_json_raises(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text("not valid json")
    with pytest.raises(AppError):
        get_prompts(f)


def test_get_prompts_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AppError):
        get_prompts(tmp_path / "nonexistent.json")


def test_get_prompts_empty_string_raises(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps([{"prompt": ""}]))
    with pytest.raises(AppError, match="Malformed"):
        get_prompts(f)


def test_get_prompts_all_empty_dicts_raises(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps([{}, {}, {}]))
    with pytest.raises(AppError, match="No inputs"):
        get_prompts(f)


def test_get_prompts_special_characters(tmp_path: Path) -> None:
    prompts = [
        {"prompt": "Run 'SELECT * FROM users' on db"},
        {"prompt": "Read C:\\Users\\john\\file.txt"},
        {"prompt": 'Say "hello world"'},
        {"prompt": "Compute √2 and return résultat"},
        {"prompt": "Use regex pattern [A-Z]{3}\\d+"},
    ]
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps(prompts))
    result = get_prompts(f)
    assert len(result) == 5
    assert result[1].prompt == "Read C:\\Users\\john\\file.txt"


def test_get_prompts_non_dict_entry_raises(tmp_path: Path) -> None:
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps(["not a dict"]))
    with pytest.raises((AppError, TypeError)):
        get_prompts(f)


# ── get_definitions ──────────────────────────────────────────────────────────


_VALID_DEF = {
    "name": "fn_add",
    "description": "Adds two numbers.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"},
}


def test_get_definitions_valid(tmp_path: Path) -> None:
    f = tmp_path / "defs.json"
    f.write_text(json.dumps([_VALID_DEF]))
    defs = get_definitions(f)
    assert len(defs) == 1
    assert defs[0].name == "fn_add"


def test_get_definitions_raw_field_populated(tmp_path: Path) -> None:
    f = tmp_path / "defs.json"
    f.write_text(json.dumps([_VALID_DEF]))
    defs = get_definitions(f)
    assert defs[0].raw == json.dumps(_VALID_DEF)


def test_get_definitions_multiple(tmp_path: Path) -> None:
    d2 = {**_VALID_DEF, "name": "fn_sub"}
    f = tmp_path / "defs.json"
    f.write_text(json.dumps([_VALID_DEF, d2]))
    defs = get_definitions(f)
    assert [d.name for d in defs] == ["fn_add", "fn_sub"]


def test_get_definitions_empty_array_raises(tmp_path: Path) -> None:
    f = tmp_path / "defs.json"
    f.write_text("[]")
    with pytest.raises(AppError, match="No function definitions"):
        get_definitions(f)


def test_get_definitions_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AppError):
        get_definitions(tmp_path / "nonexistent.json")


def test_get_definitions_invalid_json_raises(tmp_path: Path) -> None:
    f = tmp_path / "defs.json"
    f.write_text("{bad}")
    with pytest.raises(AppError):
        get_definitions(f)


def test_get_definitions_malformed_entry_raises(tmp_path: Path) -> None:
    f = tmp_path / "defs.json"
    f.write_text(
        json.dumps([{"name": "fn_no_params"}])
    )  # missing required fields
    with pytest.raises(AppError, match="Malformed"):
        get_definitions(f)


def test_get_definitions_skips_empty_dicts(tmp_path: Path) -> None:
    f = tmp_path / "defs.json"
    f.write_text(json.dumps([{}, _VALID_DEF]))
    defs = get_definitions(f)
    assert len(defs) == 1
    assert defs[0].name == "fn_add"


def test_get_definitions_no_parameters(tmp_path: Path) -> None:
    d = {**_VALID_DEF, "name": "fn_noop", "parameters": {}}
    f = tmp_path / "defs.json"
    f.write_text(json.dumps([d]))
    defs = get_definitions(f)
    assert defs[0].parameters == {}


def test_get_definitions_all_param_types(tmp_path: Path) -> None:
    d = {
        "name": "fn_mixed",
        "description": "Uses all types.",
        "parameters": {
            "text": {"type": "string"},
            "count": {"type": "number"},
            "flag": {"type": "boolean"},
        },
        "returns": {"type": "string"},
    }
    f = tmp_path / "defs.json"
    f.write_text(json.dumps([d]))
    defs = get_definitions(f)
    assert defs[0].parameters["text"].type == "string"
    assert defs[0].parameters["count"].type == "number"
    assert defs[0].parameters["flag"].type == "boolean"


def test_get_definitions_private_parameters_accepted(tmp_path: Path) -> None:
    """Parameters prefixed with __ are valid in pydantic; decoder.py filters them."""
    d = {
        **_VALID_DEF,
        "parameters": {
            "a": {"type": "number"},
            "__hidden": {"type": "string"},
        },
    }
    f = tmp_path / "defs.json"
    f.write_text(json.dumps([d]))
    defs = get_definitions(f)
    assert "__hidden" in defs[0].parameters


def test_get_definitions_unknown_param_type_accepted(tmp_path: Path) -> None:
    """Parameter type is an arbitrary string; pydantic doesn't constrain it."""
    d = {**_VALID_DEF, "parameters": {"x": {"type": "array"}}}
    f = tmp_path / "defs.json"
    f.write_text(json.dumps([d]))
    defs = get_definitions(f)
    assert defs[0].parameters["x"].type == "array"


# ── flush_results ─────────────────────────────────────────────────────────────


def test_flush_results_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    data = [{"prompt": "p", "name": "fn_test", "parameters": {"a": 1}}]
    flush_results(data, out)
    assert json.loads(out.read_text()) == data


def test_flush_results_creates_nested_directories(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "c" / "out.json"
    flush_results([], out)
    assert out.exists()


def test_flush_results_overwrites_existing(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    flush_results([{"old": True}], out)
    flush_results([{"new": True}], out)
    assert json.loads(out.read_text()) == [{"new": True}]


# ── get_files ────────────────────────────────────────────────────────────────


def _args(
    tmp_path: Path, *, input_exists: bool = True, def_exists: bool = True
) -> Namespace:
    input_f = tmp_path / "input.json"
    def_f = tmp_path / "defs.json"
    if input_exists:
        input_f.write_text("[]")
    if def_exists:
        def_f.write_text("[]")
    return Namespace(
        input=str(input_f),
        functions_definition=str(def_f),
        output=str(tmp_path / "out.json"),
    )


def test_get_files_valid(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = get_files(args)
    assert len(paths) == 3


def test_get_files_missing_input_raises(tmp_path: Path) -> None:
    args = _args(tmp_path, input_exists=False)
    with pytest.raises(AppError, match="input"):
        get_files(args)


def test_get_files_missing_definition_raises(tmp_path: Path) -> None:
    args = _args(tmp_path, def_exists=False)
    with pytest.raises(AppError, match="definition"):
        get_files(args)
