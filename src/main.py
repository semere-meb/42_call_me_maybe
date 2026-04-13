from .jsonschema import JSONSchema, States
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from string import Template

import torch

from llm_sdk import Small_LLM_Model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
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


def main():
    definition_file, input_file, output_file = parse_args()

    prompt_template = Template(f"""
    You are a function calling assistant. Given a user request, select the 
    appropriate function and extract the arguments.

    Available functions:
    {definition_file.read_text(encoding="utf-8")}

    Output JSON with keys: name, parameters.

    User request: "$request"

    Answer: 
    """)
    input_list = json.loads(input_file.read_text(encoding="utf-8"))
    model = Small_LLM_Model()

    for req in input_list[:1]:
        prompt = prompt_template.substitute(request=req["prompt"])
        input_ids = model.encode(prompt)[0].tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        schema = JSONSchema()
        response_token_ids = []
        # for _ in range(40):
        #     logits = model.get_logits_from_input_ids(input_ids)
        #     next_token_id = int(torch.tensor(logits).argmax())
        #     input_ids.append(next_token_id)
        #     response_token_ids.append(next_token_id)
        # for token_id in response_token_ids:
        #     print(repr(model.decode(token_id)))
        # text = model.decode(response_token_ids)
        # print(text)

        for _ in range(4):
            # while schema.state != States.VALID_JSON:
            logit_tensor = torch.tensor(logits)
            topk_values, topk_indices = torch.topk(logit_tensor, 1000)
            probs = torch.softmax(topk_values, dim=0).tolist()
            token_ids = topk_indices.tolist()
            tokens = [model.decode([token_id]) for token_id in token_ids]

            candidates = zip(tokens, token_ids, probs)
            # for candidate in candidates:
            #     print(
            #         f"token: {repr(candidate[0])}, token_id: {candidate[1]}, prob: {candidate[2]}"
            #     )
            valid_token_id = schema.ingest(candidates)
            # if valid_token_id is None:
            #     # TODO: handle None; increase k in topk or span over the entire vocab
            #     return
            # print("valid token id:", valid_token_id)

            input_ids.append(valid_token_id)
            response_token_ids.append(valid_token_id)

            # print("=" * 100)

        response = model.decode(response_token_ids)
        print(response)
