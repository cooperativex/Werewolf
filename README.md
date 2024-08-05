# Learning to Discuss Strategically: A Case Study on *One Night Ultimate Werewolf*
This repo contains an implementation of the *One Night Ultimate Werewolf* game environment and the RL-instructed LLM-based agent framework proposed in the paper **Learning to Discuss Strategically: A Case Study on *One Night Ultimate Werewolf***.

## Getting Start

### Step 1: Installation

Requirements:

- Python >= 3.9

For other dependencies, you can install with pip:
```bash
pip install -r requirements.txt
```

### Step 2: Setting LLM APIs
Currently, we implemented the access to ChatGPT and Gemini as the backends of agents. Before utilizing these LLMs, export your OpenAI and Gemini API key as environment variables:
```bash
export OPENAI_API_KEY="Your OpenAI API Key"
export GOOGLE_API_KEY="Your Gemini API Key"
```
or use `.env` file to set your LLM APIs in advance.

In defult, the ChatGPT model is **"gpt-4-1106-preview"** and the Gemini model is **"gemini-pro"**. If you want to change the default model, please refer to the **`DEFAULT_MODEL`** variable in corresponding backend class in `onuw/backends`.

## *One Night Ultimate Werewolf* (ONUW) Game Env
The implementation of the standard game env is in `onuw/environments/werewolf.py`, where you can modify the game logic. And other envs are implemented for experiments.

If there are new roles, you can add them in `onuw/agents/roles`, following the class structure of roles that already implemented.

## Run Game
Now the game can only be run in local cmd/bash, here is the instruction:
```bash
python main.py --env <game environment name> --num_runs <number of runs of different settings> --num_repeats <number of repeating runs in one setting> --random --cli --save_path <save path for game logs>
```
The following options are only enabled when added:
- `random`: Randomly assign roles at the beginning
- `cli`: Launch cli (the interactive interface in the command line)
