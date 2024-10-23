from typing import Dict
import os
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import IntelligenceBackend

try:
    import google.generativeai as genai
except ImportError:
    is_gemini_available = False
else:
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key is None:
        is_gemini_available = False
    else:
        is_gemini_available = True
        genai.configure(api_key=google_api_key)

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gemini-pro"


class Gemini(IntelligenceBackend):
    """
    Interface to the Gemini offered by Google.
    """
    stateful = False
    type_name = "gemini"

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS, 
                 model: str = DEFAULT_MODEL, **kwargs):
        """
        instantiate the Gemini backend
        args:
            max_tokens: the maximum number of tokens to sample
            model: the model to use
        """
        assert is_gemini_available, "google.generativeai package is not installed or the API key is not set"
        super().__init__(temperature=temperature, max_tokens=max_tokens, model=model, **kwargs)
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model

        self.client = genai.GenerativeModel(model_name=self.model)
    
    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages, *args, **kwargs):
        completion = self.client.generate_content(messages)
        response = completion.text
        return response
    
    def query(self, agent_name: str, prompts: Dict[str, str], request_msg: str = None, *args, **kwargs) -> str:
        print("Using backend with :" + DEFAULT_MODEL)
        
        # Construct the prompts for ChatGPT
        messages = [
            prompts.get("system_prompt", ""),
            prompts.get("user_prompt", "")
        ]

        # Specific action and desired JSON response format
        if request_msg:
            messages.append(request_msg)
        else:  # The default request message that reminds the agent its role and instruct it to speak
            messages.append(f"Now it is your turn, {agent_name}.")
        
        # Generate response
        response = self._get_response(messages, *args, **kwargs)

        # Post-process, remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()
        response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()

        return response
