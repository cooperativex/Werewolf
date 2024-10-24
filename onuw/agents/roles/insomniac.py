from .base import BaseRole


class Insomniac(BaseRole):
    role_name = "Insomniac"

    def __init__(self, name: str):
        super().__init__(name=name)

        self.role_description = f"""You are {self.name}, the Insomniac at the first place.
As a Insomniac, you may check your role at the end of the Night phase to see if it has changed, so you will know both your original and final role. But you won't obtain the abilities of your final role.
Your team depends on the final role you checked. If you are on Team Village, your goal is to help other teammates to find out who are the actual Werewolves, while using your infomation (such as whether be switched) to detect deceptions.
However, if you are on Team Werewolf, it means try you were switched before, so try to sow doubt and confusion among other players to avoid detection.\n""" + self.role_description
    
    def get_night_prompt(self):
        return ""
    
    def get_night_input(self):
        return {}
