import json

import numpy as np

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
vocab = json.loads(open(model.get_path_to_vocab_file()).read())
id_to_token = {id: value for value, id in vocab.items()}

prompt = "The capital of France is"
input_ids = model.encode(prompt)[0].tolist()
response_tokens = []

# for _ in range(10):
logits = model.get_logits_from_input_ids(input_ids)

# manual
# dist = list(zip(range(len(logits)), logits, strict=False))
# rank = sorted(dist, key=lambda x: x[1], reverse=True)
# next_token_id = rank[0][0]

next_token_id = int(np.array(logits).argmax())

input_ids.append(next_token_id)
next_token = model.decode([next_token_id])
response_tokens.append(next_token)

# print(prompt, end="")
# print("".join(response_tokens))

# def _main():
#     definition_file, input_file, output_file = parse_args()

#     prompt_template = Template(f"""
#     You are a function calling assistant. Given a user request, select the
#     appropriate function and extract the arguments.

#     Available functions:
#     {definition_file.read_text(encoding="utf-8")}

#     Output JSON with keys: name, parameters.

#     User request: "$request"

#     Answer:
#     """)
#     input_list = json.loads(input_file.read_text(encoding="utf-8"))
#     model = Small_LLM_Model()
#     vocab = json.loads(open(model.get_path_to_vocab_file()).read())
#     id_to_token = {id: model.decode([id]) for _, id in vocab.items()}

#     results = []
#     for req in input_list[:1]:
#         prompt = req["prompt"]
#         request_str = prompt_template.substitute(request=prompt)
#         input_ids = model.encode(request_str)[0].tolist()

#         schema = JSONSchema()
#         response_tokens = []

#         while schema.state != States.VALID_STATE:
#             logits = model.get_logits_from_input_ids(input_ids)
#             rank = numpy.array(logits).argsort()[::-1]

#             (valid_token_id, valid_token) = schema.next_token(
#                 rank, id_to_token
#             )

#             input_ids.append(valid_token_id)
#             response_tokens.append(valid_token)

#         response = json.loads("".join(response_tokens))
#         prompt_dict = {"prompt": prompt}
#         results.append({**prompt_dict, **response})
#     json.dump(results, output_file.open(mode="w"), indent=4)
