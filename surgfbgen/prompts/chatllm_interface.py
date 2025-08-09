# Note: The definitions for classes like ChatLLM, Connector, ChatLLMArgs,
# and functions like validate_llm_response_list are assumed to be imported
# from other modules as they are not fully visible in the images.
from typing import Optional, Dict, Any

class ChatLLMInterface:
    def __init__(
        self, 
        model_url: str, 
        temperature: float = None, 
        max_tokens: int = 10_000, 
        num_attempts: int = 5,
        connector_config: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> None:
        """Initialise the ChatLLM.

        Args:
            connector_config: Configuration for the Connector object
            **kwargs: Additional parameters like embedding_model_name, embedding_dim, etc.

        Returns:
            VectorDB client instance
        """
        self.chat: ChatLLM = None
        self.model_url = model_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_attempts = num_attempts

        if connector_config:
            self.connector = Connector(**connector_config)
        else:
            self.connector = Connector()

        if self.chat is None:
            self.chat = ChatLLM(connector=self.connector)

    def get_response(self, prompt: str) -> str | Any | None:
        if self.temperature is None:
            params = ChatLLMArgs.model_validate({
                "conversation": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": self.max_tokens,
            })
        else:
            params = ChatLLMArgs.model_validate({
                "conversation": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            })

        got_response = False
        response = None
        counter = 0
        while not got_response and counter < self.num_attempts:
            try:
                response = self.chat.standard_completion(self.model_url, params).response
                got_response = True
            except Exception as e:
                print(f"Error while getting request: {e}")

        # Clean up the response
        if response is not None:
            response = response.replace("```json", '').replace("```python", '').replace("```", '').strip()

        return response

    def close(self) -> None:
        """Close the connection to the chat model."""
        if self.connector:
            self.connector.close()
        if self.connector is not None:
            self.connector = None
        if self.chat is not None:
            self.chat = None