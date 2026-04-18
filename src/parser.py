import sys
from argparse import ArgumentParser
from pathlib import Path


output_default = "data/output/function_calls.json"
definition_default = "data/input/function_calling_tests.json"
input_default = "data/input/functions_definition.json"


def parse_args():
    parser = ArgumentParser(
        prog="python -m src",
        description="Performs constrained decoding from a prompt",
    )

    parser.add_argument(
        "-f",
        "--functions_definition",
        default=definition_default,
        help=f"Path to the definition file.\n\
            Default: {definition_default}",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=input_default,
        help=f"path to the prompt file.\n\
            Default: {input_default}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=output_default,
        help=f"Path to the output file.\n\
            default: {output_default}.\n\
            parent directory will be created if it doesn't exist.",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    definition_file = Path(args.functions_definition)

    if not input_file.is_file() or not definition_file.is_file():
        sys.exit()
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch()

    return definition_file, input_file, output_file
