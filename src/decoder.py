import json
from string import Template
from typing import Any

import numpy as np
from colorama import Fore

from src.errors import AppError
from src.model_wrapper import ModelWrapper
from src.models import Definition, Prompt
from src.schema import Schema


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

    try:
        matched_def = get_definition(model, definitions, prompt.prompt)
        print(
            "Definition:\n{Fore.MAGENTA}"
            f"{matched_def.model_dump_json(indent=4)}{Fore.RESET}"
        )

        params = get_parameters(model, matched_def, prompt.prompt)

        res = {
            "prompt": prompt.prompt,
            "name": matched_def.name,
            "parameters": params,
        }
        print(f"JSON:\n{Fore.MAGENTA}{json.dumps(res, indent=4)}{Fore.RESET}")
        print("=" * 50)
        return res
    except AppError:
        print(f"{Fore.RED}Did not match any function definition.")
        return {}


def get_definition(
    model: ModelWrapper, definitions: list[Definition], prompt: str
) -> Definition:
    template = Template('''
    You are a function calling assistant. Given the following list of function
    definitions and a user request, select the the name of the appropriate
    function to use.

    Function definitions:
    <definitions>
    [
    $definitions
    ]
    </definitions>

    User request:
    <prompt>
    "$prompt"
    </prompt>

    Output JSON with key: "name".

    Answer:
    <JSON>
    {"name": "''')

    name_request = template.substitute(
        definitions=", ".join(
            Definition.model_dump_json(definition)
            for definition in definitions
        ),
        prompt=prompt,
    )

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

    return definitions[function_idx]


def get_parameters(
    model: ModelWrapper, definition: Definition, prompt: str
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
    template = Template("""
    You are a function calling assistant. Given the following function
    definition and a user request, extract the COMPLETE and EXACT ARGUMENTS
    (NOT THE RESULT) from the user request to complete the JSON object.

    Function definition:
    <definition>
    $definition
    </definition>

    User request:
    <request>
    $prompt
    </request>

    Answer:
    <JSON>
    {
        "name": "$name",
        "description": "$description",
        "parameters": {
            """)
    param_request = template.substitute(
        definition=definition.model_dump_json(),
        prompt=prompt,
        name=definition.name,
        description=definition.description,
    )

    param_dict = {
        param: definition.parameters[param].type
        for param in definition.parameters
        if not param.startswith("__")
    }
    param_values: dict[str, float | int | bool | str | None] = {}

    for key, key_type in param_dict.items():
        print(f"Parameter: {Fore.BLUE}{repr(key)}{Fore.RESET}")
        param_request += f'"{key}": '

        schema = Schema(model, key_type=key_type)
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
                val_obj = val.replace('\\"', '"').replace("\\\\", "\\")
        except ValueError:
            val_obj = None

        param_values[key] = val_obj
        if key_type == "string":
            param_request += f'"{val}", '
        else:
            param_request += val + ", "
    return param_values
