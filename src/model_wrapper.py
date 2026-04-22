import json
from json import JSONDecodeError

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.errors import AppError


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
