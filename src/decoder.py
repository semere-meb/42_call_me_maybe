import json
from string import Template
from typing import Any

import numpy as np
from colorama import Fore

from src.errors import AppError
from src.model_wrapper import ModelWrapper
from src.models import Definition, Prompt
from src.schema import Schema, States


def run_prompt(
    prompt: Prompt,
    model: ModelWrapper,
    definitions: list[Definition],
) -> dict[str, Any]:
    """

    Runs the prompt, determines the function to be used and accordingly uses
    the get_prompt with the correct set of keys/parameters to construct and
    return a fully constructed object.

    Args:
      prompt: Prompt: The prompt.
      model: The model used to run the prompts.
      definitions: The list of function definitions.

    Returns:
      : The fully constructed object.

    """
    template = Template(f"""
    You are a function calling assistant. Given a user request, select the
    appropriate function and extract the arguments.

    Available functions:
    [{", ".join([definition.raw for definition in definitions])}]

    Output JSON with keys: name, parameters.

    User request: "$request"

    Answer:""")

    request = template.substitute(request=prompt.prompt)

    function_idx = get_function(request, definitions, model)
    res_dict = {"name": definitions[function_idx].name}
    name_additions = json.dumps(res_dict)[:-1]

    param_dict = {
        param: definitions[function_idx].parameters[param].type
        for param in definitions[function_idx].parameters
        if not param.startswith("__")
    }

    params = get_parameters(request + name_additions, param_dict, model)

    res = {
        "prompt": prompt.prompt,
        "name": definitions[function_idx].name,
        "parameters": params,
    }
    print(f"JSON: {Fore.MAGENTA}{json.dumps(res, indent=4)}{Fore.RESET}")
    print("=" * 50)
    return res


def get_function(
    request: str,
    definitions: list[Definition],
    model: ModelWrapper,
) -> int:
    name_request = request + '{"name": "'

    name_partial = ""
    matches = list(range(len(definitions)))

    input_ids = model.encode(name_request)
    while len(matches) > 1:
        logits = np.array(
            model.get_logits(input_ids),
            dtype=float,
        )

        for token_id in range(len(logits)):
            if token_id not in model.vocab:
                logits[token_id] = -np.inf
                continue
            token_text = model.vocab[token_id]["decoded"]
            candidate = name_partial + token_text
            if not any(
                definitions[idx].name.startswith(candidate) for idx in matches
            ):
                logits[token_id] = -np.inf

        next_token_id = int(logits.argmax())
        next_token = model.vocab[next_token_id]["decoded"]

        input_ids.append(next_token_id)
        name_partial += next_token
        matches = [
            idx
            for idx in matches
            if definitions[idx].name.startswith(name_partial)
        ]

    if len(matches) == 1:
        function_idx = matches[0]
    else:
        raise AppError("No match found.")

    return function_idx


def get_parameters(
    request: str,
    params: dict[str, str],
    model: ModelWrapper,
) -> dict[str, Any]:
    """

    Prompts and parses the result against each parameter/key and returns it
    in a dictionary.

    Args:
      request: str : The prompt string.
      params: dict[str, str] : A map of the name of a key to its 'type' in
        definition.
      model: ModelWrapper : The model used to run the prompt.
      vocab: dict[int, dict[str, str]] : The vocabulary lookup table.

    Returns:
      : dict[str, Any]: The constructed object representing the parameters
        field.


    """
    param_values: dict[str, float | int | bool | str | None] = {}
    param_request = request + ', "parameters": {'

    for key, key_type in params.items():
        print(f"Parameter: {Fore.BLUE}{repr(key)}{Fore.RESET}")
        param_request += f'"{key}": '

        init_state = States.START
        # if key_type == "string":
        #     param_request += '"'
        #     init_state = States.STR

        schema = Schema(model, init_state)
        val = schema.get_next_val(param_request)
        print(f"Value: {Fore.BLUE}{repr(val)}{Fore.RESET}")

        val_obj: float | int | bool | str | None = None
        try:
            if val == "null":
                val_obj = None
            elif key_type == "number":
                val_obj = float(val)
            elif key_type == "integer":
                val_obj = int(val)
            elif key_type == "boolean":
                val_obj = val == "true"
            else:
                val_obj = val
        except ValueError:
            val_obj = None

        param_values[key] = val_obj
        param_request += val + ", "
    return param_values
