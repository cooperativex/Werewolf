from .base import BaseRole


class Werewolf(BaseRole):
    role_name = "Werewolf"

    def __init__(self, name: str):
        super().__init__(name=name)

        self.role_description = f"""You are {self.name}, the Werewolf at the first place. 
As a Werewolf, you are on Team Werewolf and will know who are the other Werewolves. 
Your should keep your teammates' (if exists) and your role secret and avoid being detected during the Day phase, if you still consider yourself a Werewolf through the discussion.
But if you consider your role was switched to a new role on Team Village, you could reveal some other information except your initial Werewolf role to confirm your assumption.
Try to sow doubt and confusion among other players to avoid detection.\n""" + self.role_description
    
    def get_night_prompt(self):
        return ""
