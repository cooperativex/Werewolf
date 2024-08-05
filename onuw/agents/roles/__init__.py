from .base import BaseRole, SPEAKING_STRATEGY
from .villager import Villager
from .werewolf import Werewolf
from .seer import Seer
from .robber import Robber
from .troublemaker import Troublemaker
from .insomniac import Insomniac

ALL_ROLES = [
    Villager,
    Werewolf,
    Seer,
    Robber,
    Troublemaker,
    Insomniac,
]

ROLE_REGISTRY = {role.role_name: role for role in ALL_ROLES}
