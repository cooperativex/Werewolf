from .base import BaseRole


class Villager(BaseRole):
    role_name = "Villager"

    def __init__(self, name: str):
        super().__init__(name=name)

        self.role_description = f"""You are {self.name}, the Villager at the first place. 
As a Villager, you have no special abilities. However, your power comes from your ability to gather information, reason and find out who are the actual Werewolves and convince your teammates to vote them out.
Remember, you're on Team Village. Your goal is to help other teammates to find out who are the actual Werewolves.\n"""  + self.role_description
    
    def get_night_prompt(self):
        return ""
    
    def get_night_input(self):
        return {}
