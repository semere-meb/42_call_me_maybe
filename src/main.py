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

    response_token_ids = []
    for _ in range(40):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = int(torch.tensor(logits).argmax())
        input_ids.append(next_token_id)
        response_token_ids.append(next_token_id)
    text = model.decode(response_token_ids)
    print(text)
