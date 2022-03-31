from FlappyBirdGym.FlappyBirdGym import *
from FlappyBirdGym.WindowMode import *

import numpy as np


class EnvMananger:

    def __init__(self, window_mode: int):
        """Init the EnvMananger. The input of the ANN is a concatenated version of the
        current and next game state which is managed by this helper class.
        It is a wrapper around the FlappyBirdGym.

        Keyword arguments:
        window_mode -- 0 = no window, 1 = only game window, 2 = game window with plots
        """

        self.gym = FlappyBirdGym(window_mode)

        # Provide easy access to window
        if window_mode != WindowMode.NO_WINDOW:
            self.window = self.gym.window

        # env state = previous and last state of original env (gym) -> detect moving direction etc.
        current_state = self.gym.get_state()
        self.state = np.zeros(shape=(*current_state.shape, 2))
        self.state[:, 0] = current_state

    def step(self, action: int):
        """Interacts with the encapsulated environment. 
        By relaying the action the this environment.

        Keyword arguments:
        action -- action of the bird: 0 = no jump, 1 = jump

        Return:
        concatenated current and next game state, reward and "does this action end the game?"
        """
        current_state, reward, done_flag = self.gym.step(action)
        self.state[:, 0] = self.state[:, 1]
        self.state[:, 1] = current_state

        return self.state.flatten(), reward, done_flag

    def reset(self):
        """Call reset on the environment.
        Note that, current and next game state are also resetted.
        """
        self.gym.reset()

        # reset env state
        current_state = self.gym.get_state()
        self.state = np.zeros(shape=(*current_state.shape, 2))
        self.state[:, 0] = current_state

        return self.state.flatten()

    def get_state(self):
        """
        Returns the concatenated version of the current and next game state (flatten).

        Return:
        current and game state as flatten. Shape: (24,)
        """
        return self.state.flatten()

    def close(self):
        """Call close on the environment.
        """
        self.gym.close()

    @property
    def observation_space_shape(self):
        """
        Return: Return the shape of the current and next game state (flatten)
        which is (24,).
        """
        return self.state.flatten().shape

    @property
    def num_actions(self):
        """
        Return: Return the number of action of the gym which are 2 (no jump and jump)
        """
        return self.gym.num_actions
