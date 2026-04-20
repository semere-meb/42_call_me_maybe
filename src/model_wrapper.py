import json
from json import JSONDecodeError

from llm_sdk import Small_LLM_Model
from src.errors import AppError


class ModelWrapper:
    model: Small_LLM_Model
    vocab: dict[int, dict[str, str]]

    def __init__(self, model_id):
        try:
            self.model = Small_LLM_Model(model_id)
        except Exception as e:
            raise AppError(f"Can not initialize model. {e}") from e

        try:
            with open(self.model.get_path_to_vocab_file()) as vocab_file:
                vocab_raw = json.load(vocab_file)

            vocab = {
                id: {
                    "token_raw": token,
                    "token_decoded": self.model.decode([id]),
                }
                for token, id in vocab_raw.items()
            }

            self.vocab = vocab

        except OSError as e:
            raise AppError(f"Can not open vocab file. {e}") from e
        except JSONDecodeError as e:
            raise AppError(f"Can not parse vocab file. {e}") from e
