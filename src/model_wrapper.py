import json
from json import JSONDecodeError
from typing import cast

from llm_sdk import Small_LLM_Model
from src.errors import AppError


class ModelError(Exception):
    pass


class ModelWrapper:
    """

    This class bunles the model with a newly constructed vocabulary dictionary,
    which maps a token_id to a raw and decoded token.

    Attributes:
      model: Small_LLM_Model : The model instance.
      vocab: dict[int, dict[str, str]] : Vocabulary mapping from token id to
        a dict raw and decoded token.

    """

    model: Small_LLM_Model
    vocab: dict[int, dict[str, str]]

    def __init__(self, hf_model_name: str) -> None:
        try:
            self.model = Small_LLM_Model(hf_model_name)
        except Exception as e:
            raise AppError(f"Can not initialize model. {e}") from e

        try:
            with open(self.model.get_path_to_tokenizer_file()) as vocab_file:
                data = json.load(vocab_file)
                vocab_raw = data["model"]["vocab"]
                for entry in data.get("added_tokens", []):
                    vocab_raw[entry["content"]] = entry["id"]

            vocab = {
                id: {
                    "raw": token,
                    "decoded": self.model.decode([id]),
                }
                for token, id in vocab_raw.items()
            }
            self.vocab = vocab

        except OSError as e:
            raise AppError(f"Can not open vocab file. {e}") from e
        except JSONDecodeError as e:
            raise AppError(f"Can not parse vocab file. {e}") from e

    def get_logits(self, input_ids: list[int]) -> list[int | float]:
        try:
            return self.model.get_logits_from_input_ids(input_ids)
        except Exception as e:
            raise ModelError(
                f"Error: The model couldn't generate logits for the input. {e}"
            ) from e

    def encode(self, text: str) -> list[int]:
        try:
            return cast(list[int], self.model.encode(text)[0].tolist())
        except Exception as e:
            raise ModelError(
                f"Error: The text {text} couldn't be encoded by the model. {e}"
            ) from e

    def decode(self, token_id: int) -> str:
        if token_id not in self.vocab:
            raise ModelError(f"Error: {token_id} is not in the model vocab.")
        try:
            return self.vocab[token_id]["decoded"]
        except Exception as e:
            raise ModelError(
                f"Error: {token_id} couldn't be decoded by the model. {e}"
            ) from e
