import sys
from string import Template

from src.decoder import run_prompt
from src.errors import AppError
from src.io_utils import flush_results, get_definitions, get_files, get_prompts
from src.model_wrapper import ModelWrapper
from src.parser import parse_args


def main() -> None:
    """

    Orchasterate everything from parsing args to dumping the json result


    """

    args = parse_args()

    try:
        definition_path, input_path, output_path = get_files(args)

        prompts = get_prompts(input_path)
        definitions = get_definitions(definition_path)

        model_wr = ModelWrapper(args.model)
        model = model_wr.model
        vocab = model_wr.vocab
        sep = ",\n"

        prompt_template = Template(f"""
        You are a function calling assistant. Given a user request, select the
        appropriate function and extract the arguments.

        Available functions:
        [{sep.join([definition.raw for definition in definitions])}]

        Output JSON with keys: name, parameters.

        User request: "$request"

        Answer:""")
        results_all = []
        for prompt in prompts:
            result = run_prompt(
                prompt, vocab, prompt_template, model, definitions
            )
            results_all.append(result)
        flush_results(results_all, output_path)

    except AppError as e:
        print(str(e))
        sys.exit(1)
