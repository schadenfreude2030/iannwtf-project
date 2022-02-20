# import gym
from FlappyBirdGym.FlappyBirdGym import *
import numpy as np
import cv2


class EnvMananger():

    def __init__(self):
        self.env = FlappyBirdGym()  # gym.make('LunarLander-v2')#.unwrapped
        # self.env.reset()

        self.img_idx = 0
        self.depth = 3

        current_img = self.getImage()  # env.getWindowImage()
        self.state = np.zeros(shape=(*current_img.shape, self.depth))
        self.state[:, :, self.img_idx] = current_img
        self.img_idx += 1

    def step(self, action):
        img, reward, done_flag = self.env.step(action)

        self.img_idx = self.img_idx % self.depth
        self.state[:, :, self.img_idx] = self.getImage()

        self.img_idx += 1

        return self.state, reward, done_flag

    def reset(self):
        self.env.reset()

        self.img_idx = 0

        current_img = self.getImage()
        self.state = np.zeros(shape=(*current_img.shape, self.depth))
        self.state[:, :, self.img_idx] = current_img
        self.img_idx += 1

        return self.state

    def getState(self):
        return self.state

    def getImage(self):
        # Get image
        img = self.env.getWindowImage()  # render(mode="rgb_array")

        # Shrink
        img = cv2.resize(img, (225, 150))

        # Convert RGB -> Grey color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
        img = (img / 128) - 1

        return img

    def close(self):
        self.env.close()

    @property
    def observation_space_shape(self):
        return self.state.shape

    @property
    def action_space(self):
        return self.env.action_space
