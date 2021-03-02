import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np

# OpenAI Gym
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward
        """
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, rwd, done, info = self.env.step(action)
            total_reward += rwd
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # (H, W, C) -> (C, H, W)
        observation = np.transpose(observation)
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:] # TODO:?
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        ) # Compose several operations together
        observation = transforms(observation)
        return observation

def getMarioEnv():
    r"""Return an `env`. Each time Mario makes an action, the environment responds with a state.

    Returns:
        env (gym environment): it returns a state of 3D array of size (4, 84, 84) representing a 4 consecutive frames stacked state.
    """
    # Initialize Super Mario environment
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env.reset()

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    return env