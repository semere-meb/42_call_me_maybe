import json
import sys
from argparse import Namespace
from json import JSONDecodeError
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
from pydantic import ValidationError

from llm_sdk import Small_LLM_Model
from src.errors import AppError
from src.model_wrapper import ModelWrapper
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


def run_prompt(
    prompt: Prompt,
    vocab: dict[int, dict[str, str]],
    template: Template,
    model: Small_LLM_Model,
    definitions: list[Definition],
) -> str:
    request = template.substitute(request=prompt.prompt)

    name_request = request + '{"name": "'

    name_partial = ""
    match_idx = list(range(len(definitions)))

    input_ids = model.encode(name_request)[0].tolist()
    while len(match_idx) > 1:
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = int(np.array(logits).argmax())
        next_token = model.decode([next_token_id])

        input_ids.append(next_token_id)
        name_partial += next_token
        matches = [
            idx
            for idx in match_idx
            if definitions[idx].name.startswith(name_partial)
        ]
        match_idx = matches

    function_idx = 0
    if len(match_idx) == 1:
        function_idx = match_idx[0]

    res_dict = {"name": definitions[function_idx].name}
    name_additions = json.dumps(res_dict)[:-1]

    # parameter_request = request + name_additions + ', "parameters": {'

    params = {
        param: definitions[function_idx].parameters[param].type
        for param in definitions[function_idx].parameters
        if not param.startswith("__")
    }

    get_parameters(request + name_additions, params, model, vocab)

    # return str(params)
    return ""


def get_parameters(
    request: str,
    params: dict[str, str],
    model: Small_LLM_Model,
    vocab: dict[int, dict[str, str]],
) -> dict[str, Any]:
    param_values: dict[str, Any] = {}
    param_request = request + ', "parameters": {'

    for key, type in params.items():
        param_request += f'"{key}": '

        val = ""
        input_ids = model.encode(param_request)[0].tolist()

        for _ in range(5):
            logits = model.get_logits_from_input_ids(input_ids)
            next_token_id = int(np.array(logits).argmax())
            next_token_txt = model.decode([next_token_id])

            # rank = np.array(logits).argsort()[::-1]

            val += next_token_txt
            input_ids.append(next_token_id)

        param_request += val
    print(param_request)

    return param_values


def main() -> None:
    """
    Orchasterate everything from parsing args to dumping the json result
    """

    args = parse_args()

    try:
        definition_path, input_path, output_path = get_files(args)

        prompts = get_prompts(input_path)
        definitions = get_definitions(definition_path)

        model_wr = ModelWrapper(args.model)
        model = model_wr.model
        vocab = model_wr.vocab

        prompt_template = Template(f"""
        You are a function calling assistant. Given a user request, select the
        appropriate function and extract the arguments.

        Available functions:
        {"\n".join([definition.raw for definition in definitions])}

        Output JSON with keys: name, parameters.

        User request: "$request"

        Answer:""")
        for prompt in prompts[:1]:
            result = run_prompt(
                prompt, vocab, prompt_template, model, definitions
            )

    except AppError as e:
        print(str(e))
        sys.exit(1)
