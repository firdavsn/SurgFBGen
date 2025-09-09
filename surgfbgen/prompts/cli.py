"""Command-line interface for prompt management."""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from surgfbgen.prompts.base import PromptTemplate, prompt_library

def list_prompts() -> None:
    """List all available prompts."""
    prompts = prompt_library.list()

    if not prompts:
        print("No prompts found in the library.")
        return

    print(f"Found {len(prompts)} prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt['name']} (v{prompt['version']})")
        print(f"  Description: {prompt['description']}")
        # The line below was deciphered from a blurry image; it lists the parameters.
        print(f"  Parameters: {', '.join(prompt['parameters'].keys())}")

    print("\nUse 'show <prompt_name>' to view a prompt.")


def show_prompt(name: str) -> None:
    """Show a specific prompt."""
    prompt = prompt_library.get(name)

    if not prompt:
        print(f"Prompt '{name}' not found.")
        return

    print(f"Name: {prompt.name}")
    print(f"Description: {prompt.description}")
    print(f"Version: {prompt.version}")

    if prompt.metadata:
        print("Metadata:")
        for key, value in prompt.metadata.items():
            print(f"  {key}: {value}")

    print("\nTemplate:")
    print(prompt.template)
    print("-" * 40)


def add_prompt(
    template_file: str, name: str, description: Optional[str] = None
) -> None:
    """Add a new prompt from a file."""
    if not os.path.isfile(template_file):
        print(f"Error: File not found: {template_file}")
        return

    try:
        with open(template_file, "r") as f:
            template = f.read()

        prompt = PromptTemplate(
            template=template,
            name=name,
            description=description,
        )

        prompt_library.add(prompt)
        print(f"Prompt '{name}' added successfully.")

    except Exception as e:
        print(f"Error adding prompt: {e}")


def format_prompt(name: str, kwargs: List[str] = {}, verbose: bool = False) -> Optional[str]:
    """Format a prompt with the given arguments."""
    prompt = prompt_library.get(name)
    if not prompt:
        print(f"Prompt '{name}' not found.")
        return None

    try:
        result = prompt.format(**kwargs)
        if verbose:
            print("---Formatted Prompt:---")
            print(result)
            print("-" * 40)
        return result
    except Exception as e:
        print(f"Error formatting prompt: {e}")
        return None


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Prompt management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # list command
    subparsers.add_parser("list", help="List all available prompts")

    # show command
    show_parser = subparsers.add_parser("show", help="Show a specific prompt")
    show_parser.add_argument("name", help="Name of the prompt")

    # add command
    add_parser = subparsers.add_parser("add", help="Add a new prompt")
    add_parser.add_argument("name", help="Name of the prompt")
    add_parser.add_argument("template_file", help="File containing the template")
    add_parser.add_argument("-d", "--description", help="Description of the prompt")

    # format command
    format_parser = subparsers.add_parser("format", help="Format a prompt")
    format_parser.add_argument("name", help="Name of the prompt")
    format_parser.add_argument("args", nargs="*", help="Arguments for formatting (key=value)")

    # load command
    load_parser = subparsers.add_parser("load", help="Load prompts from a directory")
    load_parser.add_argument("directory", help="Directory containing prompt files")

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "list":
        list_prompts()
    elif args.command == "show":
        show_prompt(args.name)
    elif args.command == "add":
        add_prompt(args.template_file, args.name, args.description)
    elif args.command == "format":
        format_prompt(args.name, args=args.args, verbose=True)
    elif args.command == "load":
        try:
            prompt_library.load_from_directory(args.directory)
            print(f"Loaded prompts from {args.directory}")
            list_prompts()
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    # You might want to load a default directory first, e.g.
    # if os.path.exists("./prompts"):
    #     prompt_library.load_from_directory("./prompts")
    sys.exit(main())