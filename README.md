*This project has been created as part of the 42 curriculum by semebrah.*

# call me maybe

A function-calling tool that translates natural language prompts into structured JSON function calls using constrained decoding on a small language model.

---

## Description

Large language models are powerful at understanding language but unreliable at producing structured output. Left to their own devices, a 0.6B-parameter model might generate valid JSON only 30% of the time. This project achieves 100% structurally valid output by applying **constrained decoding** — modifying the model's probability distribution at every generation step so that only schema-compliant tokens can be selected.

Given a natural language prompt and a set of function definitions, the program outputs a JSON object identifying the correct function and extracting all required arguments with their correct types:

```json
{
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40, "b": 2}
}
```

The model used by default is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). The implementation works with any model supported by the bundled `llm_sdk` wrapper.

---

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
uv sync
```

This creates a virtual environment and installs all dependencies including the bundled `llm_sdk` package.

### Running

```bash
# Default paths (reads from data/input/, writes to data/output/)
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# Different model
uv run python -m src --model Qwen/Qwen3-0.6B
```

### Makefile targets

| Target | Description |
|--------|-------------|
| `make run` | Install dependencies and run with default paths |
| `make install` | Install dependencies only |
| `make lint` | Run ruff, flake8, and mypy |
| `make lint-strict` | Run mypy with `--strict` |
| `make test` | Run the pytest test suite |
| `make format` | Auto-format source with ruff |
| `make debug` | Run under pdb |
| `make clean` | Remove caches |
| `make fclean` | Remove caches and virtual environment |

### Input format

**`function_calling_tests.json`** — array of prompts:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

**`functions_definition.json`** — array of function schemas:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

Supported parameter types: `string`, `number`, `boolean`, `null`.

### Output format

**`data/output/function_calling_results.json`** — one object per prompt:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
  }
]
```

---

## Algorithm

### Overview

Generation happens in two phases for each prompt: **name selection** then **parameter extraction**. Both use constrained decoding — logits for invalid tokens are set to `-inf` before selection, making invalid output structurally impossible rather than merely unlikely.

### Phase 1 — Name selection

The prompt is formatted and the prefix `{"name": "` is appended. At each step:

1. The model produces logits over the full vocabulary (~150 000 tokens).
2. Every token whose text would make `name_so_far + token` no longer a prefix of any known function name is masked to `-inf`.
3. The highest-scoring remaining token is selected (greedy).
4. The candidate function list is narrowed. Generation stops when exactly one candidate remains.

This guarantees the output name exactly matches a defined function, character by character, regardless of what the model would have generated freely.

### Phase 2 — Parameter extraction

For each parameter, a finite automaton (FA) drives token selection according to the parameter's declared type. The FA has these states:

```
START → (quote)          → STR
      → (digit)          → NBR_PRE_DOT
      → (-)              → NBR_SIGN → NBR_PRE_DOT
      → (true|false|null) → END

STR         → (char)           → STR
            → (backslash)      → STR_ESC → (one_char) → STR
            → (str_terminator) → END

NBR_PRE_DOT → (digit)          → NBR_PRE_DOT
            → (.)              → NBR_DOT → (digit) → NBR_POST_DOT
            → (nbr_terminator) → END

NBR_POST_DOT → (digit)          → NBR_POST_DOT
             → (nbr_terminator) → END
```

At each step, the top-20 tokens by logit score are ranked. For each candidate token the automaton attempts a match:

1. **Full match** — the entire token matches a validator for the current state. The automaton transitions and the token is accepted.
2. **Split match** — the token straddles a state boundary (e.g., `)"` contains content `)` fused with the string terminator `"`). The automaton scans from the longest prefix down to find a split position where the prefix matches the current state and the suffix matches the resulting next state. Only the content portion is added to the output value.

If no split position works, the token is skipped and the next ranked token is tried.

Structural tokens (opening quote, terminating quote, commas, braces) are consumed but excluded from the extracted value. Content tokens (digits, characters, literals) are appended to the result string.

### Why a finite automaton?

Regular expressions on full token text fail when BPE tokens fuse characters from different semantic roles — a common occurrence with punctuation and delimiters. The FA handles this through the split-match mechanism: it finds the correct boundary *within* the token rather than rejecting it entirely, recovering content that would otherwise be lost.

---

## Design decisions

**`ModelWrapper` centralises SDK access.** `encode`, `decode`, and `get_logits` are delegated to a single class that also builds the vocabulary dictionary from `tokenizer.json` at startup. This includes both the BPE merge vocabulary (`["model"]["vocab"]`) and special tokens (`["added_tokens"]`), ensuring EOS and other special tokens are never missing from the lookup table.

**Top-k = 20 for parameter generation.** Scanning all ~150 000 tokens every step is wasteful; the correct token is nearly always in the top 20. If it is not, an `AppError` is raised rather than silently producing wrong output.

**Pydantic for all data models.** `Prompt`, `Definition`, and `Parameter` are validated at load time. Malformed input files produce clear error messages before the model is ever loaded.

**Structural vs content patterns.** The FA explicitly categorises each pattern as structural (quotes, terminators, whitespace) or content (digits, characters, literals). The `effective` string — the portion added to the result — is determined per-pattern rather than per-state, which correctly handles the edge case of literals (`true`, `false`, `null`) that are both content and terminators.

**`effective` is appended before the state check.** After a token is matched, its content contribution is added to `res` before checking whether the automaton has reached `END`. This ensures that the final token of a value (which both contributes content and terminates generation) is not dropped.

---

## Challenges

**BPE tokens spanning state boundaries.** The tokenizer merges character sequences freely; a token like `)"` fuses the last character of a string value with its closing quote. A naive fullmatch approach skips such tokens and relies on a cleaner alternative appearing in the top-20. The split-match mechanism resolves this correctly, scanning every split position from longest prefix to shortest until one satisfies both the current state and the resulting next state.

**Special tokens absent from the base vocabulary.** The vocabulary JSON stores BPE tokens under `["model"]["vocab"]` but special tokens (EOS, `<|im_start|>`, etc.) live separately in `["added_tokens"]`. Loading only the former caused `KeyError` crashes when the model selected an EOS token. Both sections are now merged at startup.

**Literal values transitioning to END before being recorded.** In an earlier version, the result string was only updated when the automaton stayed in a content state. Literal tokens (`true`, `false`, `null`) transition directly to `END`, so their content was silently dropped. The fix appends `effective` unconditionally before the state check.

---

## Performance

**Correctness.** JSON validity is 100% by construction: the constrained decoder cannot emit a token that violates the schema. Function name accuracy and argument accuracy depend on the model's semantic understanding; Qwen3-0.6B selects the correct function reliably on unambiguous prompts and extracts scalar string and numeric arguments well.

**Speed.** Each prompt requires one forward pass per generated token. A typical prompt with two parameters takes 20–40 forward passes and completes in 5–15 seconds on CPU. A full standard test set finishes well within the 5-minute target on standard hardware.

**Robustness.** All file I/O, model initialisation, and generation errors are caught and surfaced as `AppError` with a descriptive message. The program never crashes on malformed input — it prints the error and exits with code 1.

---

## Testing strategy

The test suite (`tests/`) uses pytest and covers all components without requiring a real model to be loaded:

**`test_schema.py`** — the FA is driven by a `_MockModel` that returns controlled logit sequences. Covers: single and multi-token string generation, integer/float/negative numbers, all three literal types, split tokens (content+terminator, structural+content, content+backslash), the skip-and-retry path when the top token is invalid, and the `max_token` safety limit.

**`test_io_utils.py`** — happy paths and all error branches for `get_prompts`, `get_definitions`, and `flush_results`. Content edge cases include: special characters in prompts (SQL, Windows paths, quoted strings, unicode, regex), all parameter types, private (`__`-prefixed) parameters, unknown parameter types, and nested output directory creation.

**`test_model_wrapper.py`** — patches `Small_LLM_Model` and provides a fake `tokenizer.json` to test vocabulary loading from both `model.vocab` and `added_tokens`, and all error paths (missing file, invalid JSON, model load failure).

**`test_models.py`** — pydantic validation for all three models, including required field enforcement and edge cases (empty prompt string, no parameters).

**`test_parser.py`** — CLI argument defaults and all four custom flag variants, using both short and long forms.

```bash
make test
# or
uv run pytest tests/ -v
```

---

## Resources

### References

- [Constrained Decoding: Grammar-Guided Generation for Structured LLM Output](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output) - Using regex and masking to generate a valid JSON with constrained decoding
- [Taking Control of LLM Outputs: An Introductory Journey into Logits](https://www.youtube.com/watch?v=EiMPQsI2__Y) - Introduction to logit manipulation, exploring sampling methods and structured generation techniques
- [Constrained beam search — HuggingFace blog](https://huggingface.co/blog/constrained-beam-search) — overview of constrained generation techniques
- [Outlines: structured text generation](https://github.com/dottxt-ai/outlines) — reference implementation of FA-based constrained decoding (not used directly; studied for design reference)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — model used by default
- [BPE tokenisation — HuggingFace NLP course](https://huggingface.co/learn/nlp-course/en/chapter6/5) — background on byte-pair encoding and why tokens can span semantic boundaries
- [tokenizers library documentation](https://huggingface.co/docs/tokenizers) — `tokenizer.json` format reference
- [pydantic v2 documentation](https://docs.pydantic.dev/) — data validation library used for all models
- [numpy documentation](https://numpy.org/doc/) — used for logit manipulation and top-k ranking

### AI usage

- **Code review** — identifying type annotation issues, and the advanced use cases of uv/uvx.
- **Test suite** — drafting the initial structure and test cases for `test_schema.py` and `test_io_utils.py`, reviewed and adjusted after reading the implementation.
- **Documentation** — this README was written with AI assistance and verified against the actual implementation.

All generated content was reviewed and understood before being accepted. The constrained decoding algorithm, finite automaton design, split-match mechanism, and overall architecture were developed and reasoned through independently.
