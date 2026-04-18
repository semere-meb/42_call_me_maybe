import json
from string import Template

import numpy

from llm_sdk import Small_LLM_Model

from .jsonschema import JSONSchema, States
from .parser import parse_args


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

    results = []
    for req in input_list[:1]:
        prompt = req["prompt"]
        request_str = prompt_template.substitute(request=prompt)
        input_ids = model.encode(request_str)[0].tolist()

        schema = JSONSchema()
        response_tokens = []

        while schema.state != States.VALID_STATE:
            logits = model.get_logits_from_input_ids(input_ids)
            rank = numpy.array(logits).argsort()[::-1]

            (valid_token_id, valid_token) = schema.next_token(
                rank, id_to_token
            )

            input_ids.append(valid_token_id)
            response_tokens.append(valid_token)

        response = json.loads("".join(response_tokens))
        prompt_dict = {"prompt": prompt}
        results.append({**prompt_dict, **response})
    json.dump(results, output_file.open(mode="w"), indent=4)
