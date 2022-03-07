from FlappyBirdGym.FlappyBirdGym import *
from FlappyBirdGym.WindowMode import *

import numpy as np


class EnvMananger():

    def __init__(self, window_mode):
        self.gym = FlappyBirdGym(window_mode)

        # Provide easy access to window
        if window_mode != WindowMode.NO_WINDOW:
            self.window = self.gym.window

            # env state = previous and last state of orginal env (gym) -> detect moving direction etc.
        current_state = self.gym.get_state()
        self.state = np.zeros(shape=(*current_state.shape, 2))
        self.state[:, 0] = current_state

    def step(self, action):
        current_state, reward, done_flag = self.gym.step(action)
        self.state[:, 0] = self.state[:, 1]
        self.state[:, 1] = current_state

        return self.state.flatten(), reward, done_flag

    def reset(self):
        self.gym.reset()

        # reset env state
        current_state = self.gym.get_state()
        self.state = np.zeros(shape=(*current_state.shape, 2))
        self.state[:, 0] = current_state

        return self.state.flatten()

    def get_state(self):
        return self.state.flatten()

    def close(self):
        self.gym.close()

    @property
    def observation_space_shape(self):
        return self.state.flatten().shape

    @property
    def num_actions(self):
        return self.gym.num_actions
