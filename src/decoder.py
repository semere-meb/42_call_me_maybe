import json
from string import Template
from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.models import Definition, Prompt
from src.schema import Schema, States


def run_prompt(
    prompt: Prompt,
    vocab: dict[int, dict[str, str]],
    template: Template,
    model: Small_LLM_Model,
    definitions: list[Definition],
) -> dict[str, Any]:
    """

    Runs the prompt, determines the function to be used and accordingly uses
    the get_prompt with the correct set of keys/parameters to construct and
    return a fully constructed object.

    Args:
      prompt: Prompt:
      vocab: dict[int, dict[str, str]]:
      template: Template:
      model: Small_LLM_Model:
      definitions: list[Definition]:

    Returns:
      : dict[str, Any] : The fully constructed object.

    """
    request = template.substitute(request=prompt.prompt)

    name_request = request + '{"name": "'

    name_partial = ""
    matches = list(range(len(definitions)))

    input_ids = model.encode(name_request)[0].tolist()
    while len(matches) > 1:
        logits = np.array(
            model.get_logits_from_input_ids(input_ids),
            dtype=float,
        )

        for token_id in range(len(logits)):
            if token_id not in vocab:
                logits[token_id] = -np.inf
                continue
            token_text = vocab[token_id]["decoded"]
            candidate = name_partial + token_text
            if not any(
                definitions[idx].name.startswith(candidate) for idx in matches
            ):
                logits[token_id] = -np.inf

        next_token_id = int(logits.argmax())
        next_token = vocab[next_token_id]["decoded"]

        input_ids.append(next_token_id)
        name_partial += next_token
        matches = [
            idx
            for idx in matches
            if definitions[idx].name.startswith(name_partial)
        ]

    function_idx = 0
    if len(matches) == 1:
        function_idx = matches[0]

    res_dict = {"name": definitions[function_idx].name}
    name_additions = json.dumps(res_dict)[:-1]

    param_dict = {
        param: definitions[function_idx].parameters[param].type
        for param in definitions[function_idx].parameters
        if not param.startswith("__")
    }

    params = get_parameters(request + name_additions, param_dict, model, vocab)

    return {
        "prompt": prompt.prompt,
        "name": definitions[function_idx].name,
        "parameters": params,
    }


def get_parameters(
    request: str,
    params: dict[str, str],
    model: Small_LLM_Model,
    vocab: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """

    Prompts and parses the result against each parameter/key and returns it
    in a dictionary.

    Args:
      request: str : The prompt string.
      params: dict[str, str] : A map of the name of a key to its 'type' in
        definition.
      model: Small_LLM_Model : The model used to run the prompt.
      vocab: dict[int, dict[str, str]] : The vocabulary lookup table.

    Returns:
      : dict[str, Any]: The constructed object representing the parameters
        field.


    """
    param_values: dict[str, Any] = {}
    param_request = request + ', "parameters": {'

    for key, key_type in params.items():
        param_request += f'"{key}": '

        if key_type == "string":
            param_request += '"'

        schema = Schema(
            model,
            vocab,
            States.STR if key_type == "string" else States.START,
        )
        val = schema.get_next_val(param_request)
        print("resutl =>", val)

        val_obj: float | int | bool | str | None = None
        try:
            if val == "null":
                val_obj = None
            elif key_type == "number":
                if "." in val:
                    val_obj = float(val)
                else:
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
