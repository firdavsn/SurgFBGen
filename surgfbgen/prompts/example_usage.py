import os
import shutil

from surgfbgen.prompts.base import PromptTemplate, prompt_library

def main():
    """A simple, step-by-step demonstration of the prompt library."""
    print("🚀 ===== Starting Simple Prompt Library Demo ===== 🚀")

    ## 1. Creating and Adding Prompts
    # We'll create two PromptTemplate objects directly and add them to the library.
    print("\nSTEP 1: Creating and adding two new prompts...")

    # Prompt 1: A simple summarizer
    summarizer_prompt = PromptTemplate(
        name="simple_summarizer",
        template="Please summarize the following text: {text_to_summarize}",
        description="A basic prompt to summarize a piece of text.",
        parameters={
            "text_to_summarize": {"description": "The text that needs summarizing."}
        }
    )

    # Prompt 2: A friendly email generator
    email_prompt = PromptTemplate(
        name="friendly_email_generator",
        template="""\
Subject: Quick Hello!

Hi {recipient_name},

Just writing to say hello and hope you are having a great week!

Best,
{sender_name}""",
        description="Generates a simple, friendly email.",
    )

    prompt_library.add(summarizer_prompt)
    prompt_library.add(email_prompt)
    print("✅ Prompts 'simple_summarizer' and 'friendly_email_generator' added.")

    ## 2. Listing All Prompts
    # Now, let's see what's in our library.
    print("\nSTEP 2: Listing all prompts in the library...")
    all_prompts = prompt_library.list()
    for prompt_info in all_prompts:
        print(f"  - Name: {prompt_info['name']} | Version: {prompt_info['version']}")
        print(f"    Description: {prompt_info['description']}")

    ## 3. Getting and Using a Specific Prompt
    # Let's retrieve the email prompt and use it.
    print("\nSTEP 3: Getting and formatting the 'friendly_email_generator' prompt...")
    retrieved_prompt = prompt_library.get("friendly_email_generator")

    if retrieved_prompt:
        formatted_email = retrieved_prompt.format(
            recipient_name="Alex",
            sender_name="Jordan"
        )
        print("--- Formatted Output ---")
        print(formatted_email)
        print("------------------------")
    else:
        print("Could not find the 'friendly_email_generator' prompt!")

    ## 4. Removing a Prompt
    # Let's remove the summarizer prompt.
    print("\nSTEP 4: Removing the 'simple_summarizer' prompt...")
    prompt_library.remove("simple_summarizer")
    print("✅ Prompt 'simple_summarizer' removed. Current prompts:")
    for prompt_info in prompt_library.list():
        print(f"  - {prompt_info['name']}") # Should only be the email prompt now

    ## 5. Saving the Library to a Directory
    # Let's save our current library (which only has the email prompt) to disk.
    save_dir = "./saved_prompts"
    print(f"\nSTEP 5: Saving the remaining prompts to the '{save_dir}' directory...")
    prompt_library.save_to_directory(save_dir)
    print(f"✅ Library saved. Check the '{save_dir}' folder for a .json file.")

    ## 6. Loading the Library from a Directory
    # To prove this works, we'll first clear the library, then load from the directory.
    print("\nSTEP 6: Clearing the library and reloading from the directory...")

    # Clear the library by removing the last prompt
    prompt_library.remove("friendly_email_generator")
    print(f"  - Library cleared. Current prompt count: {len(prompt_library.list())}")

    # Now load it back up!
    prompt_library.load_from_directory(save_dir)
    print("  - Library reloaded from disk. Final list of prompts:")
    for prompt_info in prompt_library.list():
        print(f"    - {prompt_info['name']}")

    ## 7. Cleaning Up
    # Finally, let's remove the directory we created.
    print("\nSTEP 7: Cleaning up the saved prompts directory...")
    shutil.rmtree(save_dir)
    print(f"✅ Removed '{save_dir}' directory.")

    print("\n🏁 ===== Demo Complete! ===== 🏁")


if __name__ == "__main__":
    main()