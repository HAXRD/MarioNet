import torch
import torch.nn as nn
import numpy as np
from collections import deque

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1.
        self.exploration_rate_decay = 0.99
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every= 5e5

    def act(self, state):
        """Given a state, choose an epsilon-greedy action.
        
        Arguments:
            state (3D array, int, (4, 84, 84)): a single observation of the current state, (4, 84, 84)==state_dim
        Returns:
            action_idx (int): an integer representing which action Mario will perform.
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0) # (4, 84, 84) -> (1, 4, 84, 84)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease `exploration_rate`
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, experience):
        """Add the experience to memory.
        """
        pass

    def recall(self):
        """Sample experiences from memory.
        """
        pass

    def learn(self):
        """Update online action value (Q) function with a batch of experience.
        """
        pass

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

    def cache(self, state, next_state, action, reward, done):
        """Store the experience to `self.memory` (aka replay buffer)

        Arguments:
            state ()
        """