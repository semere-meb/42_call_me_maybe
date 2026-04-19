import json
import sys
from argparse import Namespace
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from src.errors import AppError
from src.models import Definition, Prompt
from src.parser import parse_args


def get_files(args: Namespace) -> list[Path]:
    """verifies the files exist and returns paths to the files.

    Args:
      args: Namespace: The arguments.

    Returns:
        A list of Paths to the definition, input, and output files.

    Raises:
        AppError: If the input or definition file doesn't exist or cant' create
            the directory for output file.
    """

    input_path = Path(args.input)
    definition_path = Path(args.functions_definition)
    output_file = Path(args.output)

    for file in input_path, definition_path:
        if not file.is_file():
            raise AppError(
                f"{'input' if file is input_path else 'definition'} file"
                + f" {file} not found"
            )

    # reports early if the path can't be created
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise AppError(
            f"Can't create a directory for the output file {e}"
        ) from e

    return [
        definition_path,
        input_path,
        output_file,
    ]


def get_prompts(input_path: Path) -> list[Prompt]:
    """Parses and validates Prompts from the input file.

    Args:
      input_path: Path: Path to the input file.

    Returns:
        The validated list of prompts.

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
    """Parses and validates function definitions from the definition file.

    Args:
      definition_path: Path: Path to the functions definition file.

    Returns:
        The validated list of definitions.

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


def main() -> None:
    """
    Orchasterate everything from parsing args to dumping the json result
    """

    args = parse_args()

    try:
        definition_path, input_path, output_path = get_files(args)
    except AppError as e:
        print(str(e))
        sys.exit(1)

    try:
        prompts = get_prompts(input_path)
        definitions = get_definitions(definition_path)
    except AppError as e:
        print(str(e))
        sys.exit(1)
