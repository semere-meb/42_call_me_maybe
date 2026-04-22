from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    """

    Parses the command line arguments and returns a dictionary-like object
    containing input, output, and definition files' path and model name to
    use.

    Returns:
      : Namespace : A dictionary-like object containing path to the input,
        output and definition files and the name of the model to use.

    """

    output_default = "data/output/function_calling_results.json"
    definition_default = "data/input/functions_definition.json"
    input_default = "data/input/function_calling_tests.json"
    model_default = "Qwen/Qwen3-0.6B"

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
        help=f"Path to the prompt file.\n\
            Default: {input_default}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=output_default,
        help=f"Path to the output file.\n\
            Default: {output_default}.\n\
            parent directory will be created if it doesn't exist.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=model_default,
        help=f"Name of the LLM to use.\n\
            Default: {model_default}.",
    )
    return parser.parse_args()
