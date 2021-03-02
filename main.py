import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import random, datetime, os, copy

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros

from marioEnv import getMarioEnv
from agent import Mario
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

env = getMarioEnv()
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break