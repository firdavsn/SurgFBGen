import os
import base64
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import openai
import google.generativeai as genai
import backoff
import cv2
import json
import numpy as np
from datetime import datetime

import surgfbgen.config.environment

@dataclass
class SupportedModels:
    """Dataclass to hold lists of supported models by provider."""
    openai: List[str] = field(default_factory=lambda: [
        'gpt-5', 'gpt-5-mini',
        'gpt-4o', 'gpt-4o-mini',
        'gpt-4.1', 'gpt-4.1-mini'
    ])
    google: List[str] = field(default_factory=lambda: [
        'gemini-2.5-pro', 'gemini-2.5-flash'
    ])

class ChatLLMInterface:
    """
    A comprehensive interface to interact with both OpenAI and Google Gemini models
    for chat completions, supporting text and image inputs.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 1000, 
                 ChatLLM_outputs_dir: Optional[str] = os.path.join(os.environ['REPO_DIRECTORY'], 'ChatLLM_outputs')):
        """
        Initializes the chat interface for a specific model.

        Args:
            model_name (str): The name of the model to use (e.g., 'gpt-4o', 'gemini-2.5-pro').
            api_key (Optional[str]): The API key for the service. If not provided, it will
                                     try to get it from environment variables OPENAI_API_KEY or GEMINI_API_KEY.
            temperature (float): The sampling temperature for the model.
            max_tokens (int): The maximum number of tokens to generate.
            ChatLLM_outputs_dir (Optional[str]): If provided, saves all LLM inputs and outputs to this directory as JSON files.
        """
        self.supported_models = SupportedModels()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.outputs_dir = os.path.join(ChatLLM_outputs_dir, model_name, datetime.now().strftime("%Y%m%d_%H%M%S")) if ChatLLM_outputs_dir else None
        
        # Create the output directory if it's specified and doesn't exist
        if self.outputs_dir:
            os.makedirs(self.outputs_dir, exist_ok=True)
            
        # Validate model name and set provider immediately
        self.provider = self._get_provider(model_name)
        
        self.client = self._initialize_client(api_key)


    def _get_provider(self, model_name: str) -> str:
        """Determines the provider (openai or google) based on the model name."""
        all_supported_openai = self.supported_models.openai
        all_supported_google = self.supported_models.google
        
        if model_name in all_supported_openai:
            return "openai"
        elif model_name in all_supported_google:
            return "google"
        else:
            raise ValueError(f"Model '{model_name}' is not supported. "
                             f"Supported OpenAI models: {all_supported_openai}. "
                             f"Supported Google models: {all_supported_google}.")

    def _initialize_client(self, api_key: Optional[str]) -> Any:
        """Initializes the API client for the determined provider."""
        if self.provider == "openai":
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided or found in environment variables.")
            return openai.OpenAI(api_key=api_key)
        elif self.provider == "google":
            if api_key is None:
                api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Google API key not provided or found in environment variables.")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model_name)
        return None

    def _format_user_prompt(self, user_prompt: Union[str, list, tuple]) -> List[Dict[str, Any]]:
        """
        Formats the user prompt to handle text, images, or a mix of both,
        preserving the chronological order of the input.
        Images can be file paths or numpy arrays (frames).

        Args:
            user_prompt (Union[str, list, tuple]): The user's prompt. Can be a string,
                                                   a list/tuple of images, or a list/tuple
                                                   containing a mix of strings and images.

        Returns:
            List[Dict[str, Any]]: A list of content parts for the API request.
        """
        # Handle the simple case of a text-only prompt for efficiency.
        if isinstance(user_prompt, str):
            if self.provider == 'openai':
                return [{"type": "text", "text": user_prompt}]
            else: # google
                return [{"text": user_prompt}]

        content_parts = []
        # Ensure user_prompt is a list or tuple to iterate
        prompt_items = list(user_prompt) if isinstance(user_prompt, (list, tuple)) else [user_prompt]

        # Iterate through the prompt items once to preserve order.
        for item in prompt_items:
            # Check if the item is a text string (and not a valid file path).
            if isinstance(item, str) and not os.path.exists(item):
                if self.provider == 'openai':
                    content_parts.append({"type": "text", "text": item})
                else: # google
                    content_parts.append({"text": item})

            # Otherwise, treat it as an image (either a file path or a numpy array).
            else:
                try:
                    # Load image from path or use the numpy array directly.
                    if isinstance(item, str):
                        img = cv2.imread(item)
                        if img is None:
                            print(f"Warning: Could not read image from path: {item}")
                            continue
                    elif isinstance(item, np.ndarray):
                        img = item
                    else:
                        print(f"Warning: Skipping unsupported item type: {type(item)}")
                        continue

                    # Encode the image to base64.
                    _, img_encoded = cv2.imencode('.jpg', img)
                    base64_image = base64.b64encode(img_encoded).decode('utf-8')

                    # Append the formatted image part based on the provider.
                    if self.provider == "openai":
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        })
                    elif self.provider == "google":
                        content_parts.append({
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        })

                except Exception as e:
                    print(f"Warning: Failed to process image item. Error: {e}")

        return content_parts

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, ConnectionResetError, json.decoder.JSONDecodeError), max_tries=5)
    def _generate_openai(self, messages: List[Dict[str, Any]]) -> str:
        """Handles chat completions for OpenAI with backoff."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    @backoff.on_exception(backoff.expo, (Exception), max_tries=5) # genai has more generic exceptions
    def _generate_google(self, contents: List[Any], system_prompt: str) -> str:
        """Handles chat completions for Google Gemini with backoff."""
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        # Gemini API prefers system instruction separately for newer models
        model_instance = genai.GenerativeModel(
            self.model_name,
            system_instruction=system_prompt,
            generation_config=generation_config
        )
        response = model_instance.generate_content(contents)
        
        try:
            return response.text
        except ValueError:
            finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
            return f"Content generation failed for Google Gemini. Finish reason: {finish_reason.name if hasattr(finish_reason, 'name') else finish_reason}."

    def _save_output_to_json(self, system_prompt: str, formatted_user_content: List[Dict[str, Any]], response: str):
        """Saves the LLM interaction to a JSON file if an output directory is configured."""
        if not self.outputs_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Sanitize model name for filename
        safe_model_name = self.model_name.replace('.', '_')
        filename = f"{timestamp}_{self.provider}_{safe_model_name}.json"
        filepath = os.path.join(self.outputs_dir, filename)

        output_data = {
            "model_details": {
                "provider": self.provider,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            "timestamp_utc": datetime.utcnow().isoformat(),
            "inputs": {
                "system_prompt": system_prompt,
                "user_content": formatted_user_content
            },
            "output": {
                "response_text": response
            }
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save LLM output to {filepath}. Error: {e}")


    def generate(self, user_prompt: Union[str, list, tuple], system_prompt: str = 'You are a helpful assistant who performs the specified task well without hallucinating.') -> str:
        """
        Generates a response from the model and saves the interaction to a JSON file if configured.

        Args:
            user_prompt (Union[str, list, tuple]): The user's prompt, which can be text,
                                                   a list of image paths/frames, or a mix.
            system_prompt (str): The instruction or context for the model.

        Returns:
            str: The generated text response from the model.
        """
        formatted_user_content = self._format_user_prompt(user_prompt)
        response_text = ""

        if self.provider == "openai":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_content}
            ]
            response_text = self._generate_openai(messages)

        elif self.provider == "google":
            response_text = self._generate_google(formatted_user_content, system_prompt)
        
        # Save the input and output to a JSON file if a directory is specified
        self._save_output_to_json(system_prompt, formatted_user_content, response_text)

        return response_text

# --- Example Usage ---
if __name__ == '__main__':
    # Make sure to set your API keys as environment variables:
    # export OPENAI_API_KEY='your_openai_key'
    # export GEMINI_API_KEY='your_google_key'

    # --- Text-only Example ---
    print("--- 1. OpenAI Text-only Example ---")
    try:
        openai_interface = ChatLLMInterface(model_name='gpt-4o')
        system_p = "You are a helpful assistant."
        user_p = "What is the capital of France?"
        # Corrected argument order
        response = openai_interface.generate(user_prompt=user_p, system_prompt=system_p)
        print(f"User: {user_p}\nAI: {response}\n")
    except (ValueError, openai.AuthenticationError) as e:
        print(f"Error: {e}\n")


    print("--- 2. Google Gemini Text-only Example ---")
    try:
        gemini_interface = ChatLLMInterface(model_name='gemini-2.5-flash', max_tokens=10000)
        system_p = "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
        user_p = "What is recursion?"
        # Corrected argument order
        response = gemini_interface.generate(user_prompt=user_p, system_prompt=system_p)
        print(f"User: {user_p}\nAI: {response}\n")
    except (ValueError) as e: # Catching generic value error for key issues
        print(f"Error: {e}\n")
    
    # --- Image Example ---
    try:
        import numpy as np
        # Create a dummy image file for testing
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_image[:, :50] = [255, 0, 0]  # Blue half
        dummy_image[:, 50:] = [0, 0, 255]  # Red half
        cv2.imwrite("test_image.jpg", dummy_image)
        
        print("--- 3. OpenAI Image Example ---")
        openai_vision_interface = ChatLLMInterface(model_name='gpt-4o')
        system_p_vision = "You are an image analysis expert."
        user_p_vision = ["Describe the colors in this image.", "test_image.jpg"]
        # Corrected argument order
        response = openai_vision_interface.generate(user_prompt=user_p_vision, system_prompt=system_p_vision)
        print(f"User: Describe the colors in this image.\nAI: {response}\n")

        print("--- 4. Google Gemini Image Example ---")
        gemini_vision_interface = ChatLLMInterface(model_name='gemini-2.5-flash')
        # Gemini can take the image as a numpy array directly
        user_p_vision_gemini = ("What is this image composed of?", dummy_image)
        # Corrected argument order
        response = gemini_vision_interface.generate(user_prompt=user_p_vision_gemini, system_prompt=system_p_vision)
        print(f"User: What is this image composed of?\nAI: {response}\n")

    except ImportError:
        print("\nPlease install numpy and opencv-python (`pip install numpy opencv-python`) to run the image examples.")
    except Exception as e:
        print(f"\nAn error occurred during the image example: {e}")

    print("--- 5. Unsupported Model Example ---")
    try:
        # This will raise a ValueError immediately
        unsupported_interface = ChatLLMInterface(model_name='claude-3')
    except ValueError as e:
        print(f"Successfully caught expected error: {e}\n")

    print("--- 6. OpenAI with Output Saving Example ---")
    try:
        # Define a directory to store outputs
        output_dir = "llm_runs"
        
        # Pass the directory to the interface
        openai_interface_save = ChatLLMInterface(
            model_name='gpt-4o',
            ChatLLM_outputs_dir=output_dir
        )
        system_p = "You are a historian."
        user_p = "Briefly, what was the significance of the printing press?"
        
        print(f"Generating response and saving to '{output_dir}' directory...")
        response = openai_interface_save.generate(user_prompt=user_p, system_prompt=system_p)
        print(f"User: {user_p}\nAI: {response}\n")
        
        # Verify a file was created
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"Successfully saved output to a JSON file in '{output_dir}'.")
            # Clean up the created directory for tidiness
            # import shutil
            # shutil.rmtree(output_dir)
            # print(f"Cleaned up '{output_dir}' directory.")
        else:
            print("Error: Output file was not created.")

    except (ValueError, openai.AuthenticationError) as e:
        print(f"Error: {e}\n")