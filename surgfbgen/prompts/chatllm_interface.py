import os
import base64
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import openai
import google.generativeai as genai
import backoff
import cv2
import json

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

    def __init__(self, model_name: str, api_key: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 1000):
        """
        Initializes the chat interface for a specific model.

        Args:
            model_name (str): The name of the model to use (e.g., 'gpt-4o', 'gemini-2.5-pro').
            api_key (Optional[str]): The API key for the service. If not provided, it will
                                     try to get it from environment variables OPENAI_API_KEY or GEMINI_API_KEY.
            temperature (float): The sampling temperature for the model.
            max_tokens (int): The maximum number of tokens to generate.
        """
        self.supported_models = SupportedModels()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
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
        Formats the user prompt to handle text, images, or a mix of both.
        Images can be file paths or numpy arrays (frames).

        Args:
            user_prompt (Union[str, list, tuple]): The user's prompt. Can be a string,
                                                   a list/tuple of images, or a list/tuple
                                                   containing a string and images.

        Returns:
            List[Dict[str, Any]]: A list of content parts for the API request.
        """
        content_parts = []
        if isinstance(user_prompt, str):
            if self.provider == 'openai':
                content_parts.append({"type": "text", "text": user_prompt})
            else: # google
                content_parts.append({"text": user_prompt})
            return content_parts

        prompt_items = list(user_prompt)
        text_part = ""

        # Extract text from the list
        text_items = [item for item in prompt_items if isinstance(item, str) and not os.path.exists(item)]
        if text_items:
            text_part = "\n".join(text_items)
            if self.provider == 'openai':
                content_parts.append({"type": "text", "text": text_part})
            else:
                content_parts.append({"text": text_part})


        # Process images
        for item in prompt_items:
            if not isinstance(item, str) or (isinstance(item, str) and os.path.exists(item)): # It's an image path or numpy array
                try:
                    if isinstance(item, str): # It's a file path
                        img = cv2.imread(item)
                        if img is None:
                            print(f"Warning: Could not read image from path: {item}")
                            continue
                    else: # Assumes it's a numpy array (frame)
                        img = item

                    _, img_encoded = cv2.imencode('.jpg', img)
                    base64_image = base64.b64encode(img_encoded).decode('utf-8')
                    
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
                    print(f"Warning: Failed to process image item '{item}'. Error: {e}")

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
        
        # FIX: Check if the response has parts before accessing .text
        try:
            return response.text
        except ValueError:
            # If response.text fails, return a more informative message
            finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
            return f"Content generation failed for Google Gemini. Finish reason: {finish_reason.name if hasattr(finish_reason, 'name') else finish_reason}."


    def generate(self, system_prompt: str, user_prompt: Union[str, list, tuple]) -> str:
        """
        Generates a response from the model.

        Args:
            system_prompt (str): The instruction or context for the model.
            user_prompt (Union[str, list, tuple]): The user's prompt, which can be text,
                                                   a list of image paths/frames, or a mix.

        Returns:
            str: The generated text response from the model.
        """
        formatted_user_content = self._format_user_prompt(user_prompt)

        if self.provider == "openai":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_content}
            ]
            return self._generate_openai(messages)

        elif self.provider == "google":
            # For Google, the 'contents' is just the user part. System prompt is handled separately.
            return self._generate_google(formatted_user_content, system_prompt)
        
        return "" # Should not be reached

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
        response = openai_interface.generate(system_p, user_p)
        print(f"User: {user_p}\nAI: {response}\n")
    except (ValueError, openai.AuthenticationError) as e:
        print(f"Error: {e}\n")


    print("--- 2. Google Gemini Text-only Example ---")
    try:
        gemini_interface = ChatLLMInterface(model_name='gemini-2.5-flash', max_tokens=10000)
        system_p = "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
        user_p = "What is recursion?"
        response = gemini_interface.generate(system_p, user_p)
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
        response = openai_vision_interface.generate(system_p_vision, user_p_vision)
        print(f"User: Describe the colors in this image.\nAI: {response}\n")

        print("--- 4. Google Gemini Image Example ---")
        gemini_vision_interface = ChatLLMInterface(model_name='gemini-2.5-flash')
        # Gemini can take the image as a numpy array directly
        user_p_vision_gemini = ("What is this image composed of?", dummy_image)
        response = gemini_vision_interface.generate(system_p_vision, user_p_vision_gemini)
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