import json
from argparse import Namespace
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.errors import AppError
from src.models import Definition, Prompt


def get_files(args: Namespace) -> list[Path]:
    """

    verifies the files exist and returns paths to the files.

    Args:
      args: Namespace: The arguments.

    Returns:
      : A list of Paths to the definition, input, and output files.

    Raises:
      AppError: If the input or definition file doesn't exist or cant' create
    the directory for output file.

    """

    input_path = Path(args.input)
    definition_path = Path(args.functions_definition)
    output_path = Path(args.output)

    for file in input_path, definition_path:
        if not file.is_file():
            raise AppError(
                f"{'input' if file is input_path else 'definition'} file"
                + f" {file} not found"
            )

    return [
        definition_path,
        input_path,
        output_path,
    ]


def get_prompts(input_path: Path) -> list[Prompt]:
    """

    Parses and validates Prompts from the input file.

    Args:
      input_path: Path: Path to the input file.
      input_path: Path:
      input_path: Path:

    Returns:
      : The validated list of prompts.

    Raises:
      AppError: If the file has issue opening and/or reading,
    empty or malformed input/prompt.

    """

    try:
        with open(input_path) as input_file:
            prompts_list = json.load(input_file)

            prompts = []
            for prompt_dict in prompts_list:
                if not prompt_dict:
                    continue
                try:
                    prompt = Prompt(**prompt_dict)
                except ValidationError as e:
                    raise AppError(f"Malformed input: {prompt_dict}") from e
                else:
                    prompts.append(prompt)
            if not prompts:
                raise AppError("No inputs were provided")
            return prompts

    except OSError as e:
        raise AppError(f"Can not open input file. {e}") from e
    except JSONDecodeError as e:
        raise AppError(
            f"Input file '{input_path}' is not a valid json file. " + str(e)
        ) from e


def get_definitions(definition_path: Path) -> list[Definition]:
    """

    Parses and validates function definitions from the definition file.

    Args:
      definition_path: Path: Path to the functions definition file.
      definition_path: Path:
      definition_path: Path:

    Returns:
      : The validated list of definitions.

    Raises:
      AppError: If the file has issue opening and/or reading,
    empty or malformed definition.

    """
    try:
        with open(definition_path) as definition_file:
            definitions_list = json.load(definition_file)

            definitions = []
            for definition_dict in definitions_list:
                if not definition_dict:
                    continue
                try:
                    definition = Definition(**definition_dict)
                except ValidationError as e:
                    raise AppError(
                        f"Malformed function definition: {definition_dict}"
                    ) from e
                else:
                    definition.raw = json.dumps(definition_dict)
                    definitions.append(definition)
            if not definitions:
                raise AppError("No function definitions were provided")

            return definitions

    except OSError as e:
        raise AppError(f"Can not open function definition file. {e}") from e
    except JSONDecodeError as e:
        raise AppError(
            f"Function definition file '{definition_path}' is not a valid json"
            + "file. "
            + str(e)
        ) from e


def flush_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """

    Args:
      results: list[dict]: List of json results for each prompt.
      output_path: Path: Output file location.

    Raises:
        AppError: If output dir/file can't be created or if there was an issue
            with writing to the file, if there was an issue encoding to json
            i.e. non-serilizable object.


    """

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=4)
    except OSError as e:
        raise AppError(f"Can not open output file. {e}") from e
    except TypeError as e:
        raise AppError(
            f"There was an issue saving output to '{output_file}'. " + str(e)
        ) from e
