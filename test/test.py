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
