from argparse import ArgumentParser, Namespace

output_default = "data/output/function_calls.json"
definition_default = "data/input/functions_definition.json"
input_default = "data/input/function_calling_tests.json"


def parse_args() -> Namespace:
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
    return parser.parse_args()
