import os
import time
import argparse
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from onuw.arena import Arena

ENV_CONFIGS = {
    "Werewolf": "werewolf.json",  # standard 5-player game
    "WerewolfEasy": "werewolf_easy.json",  # fixed 5-player game in the easy setting
    "WerewolfHard": "werewolf_hard.json",  # fixed 5-player game in the hard setting
    "Werewolf3P": "werewolf_3p.json",  # 3-player game
    "Werewolf3PWO": "werewolf_3p_wo.json"  # 3-player game without discussion
}


def main(args):
    config_path = os.path.join("configs", ENV_CONFIGS[args.env])
    arena = Arena.from_config(config_path, randomness=args.random)
    
    config = arena.to_config()
    print(config.environment)
    model_list = [player['backend']['model'] for player in config["players"]]

    model_name_combinations = []
    unique_model_names = set(model_list)
    if len(unique_model_names) == 1:  # All models have the same name
        model_name = "ALL[" + unique_model_names.pop() + "]" 
    else:
        for idx, model in enumerate(model_list):
            model_name_combinations.append(f"p{idx + 1}[{model}]")
            model_name = "_".join(model_name_combinations)
    print("Model Name:", model_name)

    for j in range(args.num_repeats):
        print(f"Repeat run {j+1} begins.")
        if args.cli:
            arena.launch_cli(interactive=True)
        else:
            arena.reset()
            arena.run(num_steps=30)

        if args.save_path:  # save history
            cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            save_dir = os.path.join(os.getcwd(), args.save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            history_address = os.path.join(save_dir, f"{model_name}_{cur_time}.json")
            arena.save_history(history_address)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Werewolf", help="choose the environment")
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs of different settings")
    parser.add_argument("--num_repeats", type=int, default=1, help="number of repeating runs in one setting")
    parser.add_argument("--random", action="store_true", default=False, help="whether to randomly assign roles at the beginning")
    parser.add_argument("--cli", action="store_true", default=False, help="whether to launch cli")
    parser.add_argument("--save_path", type=str, help="save path for game results")
    args = parser.parse_args()

    for i in range(args.num_runs):
        print(f"Run {i+1} begins.")
        main(args)
