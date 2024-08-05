from typing import List, Dict
import os
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import IntelligenceBackend
from ..memory import Message

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
    
    def query(self, agent_name: str, role_desc: str, history_messages: List[Message], global_prompt: str = None,
              request_msg: str = None, *args, **kwargs) -> str:
        """
        format the input and call the Gemini API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """
        print("Using backend with :" + DEFAULT_MODEL)
        
        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"You are a good conversation game player.\n{global_prompt.strip()}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
        else:
            system_prompt = f"You are a good conversation game player. Your name is {agent_name}.\n\nYour role:{role_desc}"

        # Concatenate conversations
        conversation_history = ""
        for msg in history_messages:
            conversation_history = f"{conversation_history}\n[{msg.agent_name}]: {msg.content}"
        
        # Instructions for different phases
        if "Night" in kwargs.get("current_phase"):
            user_prompt = f"""Now it is the Night phase. Notice that you are {agent_name}. 
Based on the game rules, role descriptions and your experience, think about your acting strategy and take a proper action."""
        
        elif "Day" in kwargs.get("current_phase"):
            user_prompt = f"""Now it is the Day phase. Here are some conversation history you can refer to: {conversation_history}
Notice that you are {agent_name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
And here is your belief about possible roles of all players: {kwargs.get("current_belief", "")}
Based on the game rules, role descriptions, messages and your belief, think about what insights you can summarize from the conversation and your speaking strategy next.
After that, give a concise but informative and specific public speech besed on your insights and strategy."""
        
        elif "Voting" in kwargs.get("current_phase"):
            user_prompt = f"""Now it is the Voting phase. Here are some conversation history you can refer to: {conversation_history}
Notice that you are {agent_name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
And here is your belief about possible roles of all players: {kwargs.get("current_belief", "")}
Based on the game rules, role descriptions, messages and your belief, think about who is most likely a Werewolf and then vote for this player."""
        
        elif "Belief" in kwargs.get("current_phase"):
            user_prompt = f"""Here are some conversation history you can refer to: {conversation_history}
Notice that you are {agent_name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
Based on the game rules, role descriptions and messages, think about what roles all players (including yourself) can most probably be now."""
        
        elif "Strategy" in kwargs.get("current_phase"):
            user_prompt = f"""Now it is the Day phase. Here are some conversation history you can refer to: {conversation_history}
Notice that you are {agent_name} in the conversation. You should carefully analyze the conversation history since some ones might deceive during the conversation.
And here is your belief about possible roles of all players: {kwargs.get("current_belief", "")}
Based on the game rules, role descriptions, messages and your belief, think about what kind of speaking strategy you are going to use for your upcoming speech in this turn"""
        
        else:
            user_prompt = ""
        
        # Construct the prompts for ChatGPT
        messages = [
            system_prompt,
            user_prompt
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
    
    def new_query(self, agent_name: str, prompts: Dict[str, str], request_msg: str = None, *args, **kwargs) -> str:
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
