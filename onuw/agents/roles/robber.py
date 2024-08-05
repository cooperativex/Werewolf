from .base import BaseRole


class Robber(BaseRole):
    role_name = "Robber"

    def __init__(self, name: str):
        super().__init__(name=name)

        self.role_description = f"""You are {self.name}, the Robber at the first place. 
As a Robber, you may choose to switch roles with another player and then become the new role you switched, but you won't obtain the abilities of your new role.
The player who you switched with becomes the new Robber and is also on Team Village. But your team will depends on what new role you got.
If you choose not to switch with another player, you remain the Robber and you are still on Team Village.\n""" + self.role_description
    
    def get_night_prompt(self):
        action_prompt = f"""Now it is your turn, {self.name}.
Please think about your acting strategy and choose whether to switch roles with another player. If switch, please give the player you want to switch with. 
You can only choose from the following options: [{', '.join(self.current_players)}].
You must return your response in a JSON format that can be parsed by Python `json.loads`. Here is the Response Format:
{{
    "thought": <your acting strategy and the reason why you act in this way>,
    "switch": <`true` or `false`, whether to switch roles with another player>,
    "player": <the player to switch with>
}}
"""
        return action_prompt
