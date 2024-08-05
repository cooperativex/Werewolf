from .base import Environment, TimeStep
from .werewolf import Werewolf
from .werewolf_easy import WerewolfEasy
from .werewolf_hard import WerewolfHard
from .werewolf_3p import Werewolf3P
from .werewolf_3p_wo import Werewolf3PWO

from ..config import EnvironmentConfig

ALL_ENVIRONMENTS = [
    Werewolf,
    WerewolfEasy,
    WerewolfHard,
    Werewolf3P,
    Werewolf3PWO,
]

ENV_REGISTRY = {env.type_name: env for env in ALL_ENVIRONMENTS}


# Load an environment from a config dictionary
def load_environment(config: EnvironmentConfig):
    try:
        env_cls = ENV_REGISTRY[config["env_type"]]
    except KeyError:
        raise ValueError(f"Unknown environment type: {config['env_type']}")

    env = env_cls.from_config(config)
    return env
