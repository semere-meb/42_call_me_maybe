import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from string import Template

import numpy

from llm_sdk import Small_LLM_Model

from .jsonschema import JSONSchema, States


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
    vocab = json.loads(open(model.get_path_to_vocab_file()).read())
    id_to_token = {id: model.decode([id]) for _, id in vocab.items()}

    for req in input_list[8:9]:
        prompt = prompt_template.substitute(request=req["prompt"])
        input_ids = model.encode(prompt)[0].tolist()

        schema = JSONSchema()
        response_tokens = []

        while schema.state != States.VALID_STATE:
            logits = model.get_logits_from_input_ids(input_ids)
            rank = numpy.array(logits).argsort()[::-1]

            (valid_token_id, valid_token) = schema.ingest(rank, id_to_token)
            print(valid_token, schema.state, schema.stack)
            print("=" * 50)

            input_ids.append(valid_token_id)
            response_tokens.append(valid_token)

        print("".join(response_tokens))
