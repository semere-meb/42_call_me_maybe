import os
from argparse import Namespace
from pathlib import Path

import pytest

from src.errors import AppError
from src.main import get_definitions, get_files, get_prompts


# get_files
def test_get_files_raises_on_not_file(tmp_path):
    args = Namespace(
        input=tmp_path,
        functions_definition=tmp_path / "definition_file.json",
        output=tmp_path,
    )

    with pytest.raises(AppError):
        get_files(args)


def test_get_files_creates_output_directory(tmp_path):
    input_file = tmp_path / "input.json"
    definition_file = tmp_path / "definitions.json"
    input_file.touch()
    definition_file.touch()

    args = Namespace(
        input=str(input_file),
        functions_definition=str(definition_file),
        output=str(tmp_path / "subdir" / "out.json"),
    )
    _ = get_files(args)
    assert (tmp_path / "subdir").is_dir()


@pytest.mark.skipif(os.getuid() == 0, reason="root ignores file permissions")
def test_get_files_raises_on_unwritable_output_dir(tmp_path):
    input_file = tmp_path / "input.json"
    definition_file = tmp_path / "definitions.json"
    input_file.touch()
    definition_file.touch()

    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    locked_dir.chmod(0o444)

    args = Namespace(
        input=str(input_file),
        functions_definition=str(definition_file),
        output=str(locked_dir / "subdir" / "out.json"),
    )
    with pytest.raises(AppError):
        get_files(args)


# get_prompts
def test_get_promts_raises_on_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{invalid_key: 1}")

    with pytest.raises(AppError):
        get_prompts(bad_file)


def test_get_prompts_raises_on_missing_input():
    with pytest.raises(AppError):
        get_prompts(Path("nonexistent.json"))


def test_get_prompts_raises_on_permission_error(tmp_path):
    locked_file = tmp_path / "locked.json"
    locked_file.write_text('[{"prompt": "hello"}]')
    locked_file.chmod(0o000)

    with pytest.raises(AppError):
        get_prompts(locked_file)


def test_get_prompts_raises_on_empty_inputs(tmp_path):
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("{}")

    with pytest.raises(AppError):
        get_prompts(empty_file)


def test_get_prompts_returns_valid_prompts(tmp_path):
    good_file = tmp_path / "good.json"
    good_file.write_text('[{"prompt": "hello"}]')

    result = get_prompts(good_file)
    assert len(result) == 1
    assert result[0].prompt == "hello"


# get_definitions
def test_get_definitions_raises_on_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{invalid_key: 1}")

    with pytest.raises(AppError):
        get_definitions(bad_file)


def test_get_definitions_raises_on_missing_input(tmp_path):
    missing_file = tmp_path / "missing.json"

    with pytest.raises(AppError):
        get_definitions(missing_file)


def test_get_definitions_raises_on_permission_error(tmp_path):
    locked_file = tmp_path / "locked.json"
    locked_file.write_text('[{"prompt": "hello"}]')
    locked_file.chmod(0o000)

    with pytest.raises(AppError):
        get_definitions(locked_file)


def test_get_definitions_raises_on_empty_inputs(tmp_path):
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("{}")

    with pytest.raises(AppError):
        get_definitions(empty_file)


def test_get_definitions_returns_valid_prompts(tmp_path):
    good_file = tmp_path / "good.json"
    good_file.write_text(r"""[{
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  }]""")

    result = get_definitions(good_file)
    assert len(result) == 1
    assert result[0].name == "fn_add_numbers"
