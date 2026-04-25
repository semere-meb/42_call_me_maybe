import sys

from colorama import Fore

from src.decoder import run_prompt
from src.errors import AppError
from src.io_utils import flush_results, get_definitions, get_files, get_prompts
from src.model_wrapper import ModelError, ModelWrapper
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

        model = ModelWrapper(args.model)

        results_all = []
        for prompt in prompts[7:9]:
            print(f"prompt: {Fore.GREEN}{repr(prompt.prompt)}{Fore.RESET}")
            result = run_prompt(prompt, model, definitions)
            results_all.append(result)
        flush_results(results_all, output_path)

    except (AppError, ModelError) as e:
        print(str(e))
        sys.exit(1)
