from .base import BaseRole
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style


class Seer(BaseRole):
    role_name = "Seer"

    def __init__(self, name: str):
        super().__init__(name=name)

        self.role_description = f"""You are {self.name}, the Seer at the first place. 
As a Seer, you may check another player's role, or two roles in the `role pool` during the Night phase.
However, the player's role you saw may be changed by other roles acting after you, so you should make a good and careful use of this information during the Day phase.
Remember, you are on Team Village. Your goal is to help other teammates to find out who are the actual Werewolves.\n""" + self.role_description

    def get_night_prompt(self):
        action_prompt = f"""Now it is your turn, {self.name}.
Please think about your acting strategy and choose which player's role you want to check or to check two roles in the `role pool` (where contains roles that are not assigned to players). 
You can only choose one from the following options: [{', '.join(self.current_players)}, role pool].
You must return your response in a JSON format that can be parsed by Python `json.loads`. Here is the Response Format:
{{
    "thought": <your acting strategy and the reason why you check this player>,
    "player": <the player to check>
}}
"""
        return action_prompt
    
    def get_night_input(self):
        player = prompt(
            [('class:info', "[Info] "),
             ('class:info_text', """As a Seer, you may check another player's role, or two roles in the `role pool`.
You can only choose one from the following options: """),
             ('class:choices', f"{', '.join(self.current_players)}, role pool.\n"),
             ('class:user_prompt', "Type the player (or the `role pool`) you want to check: ")],
            style=Style.from_dict({'info': "red bold", 'choices': "orange bold", 'user_prompt': 'ansicyan underline'})
        )
        return {"thought": "", "player": player}
