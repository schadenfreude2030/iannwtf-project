#import gym
from FlappyBirdGym.FlappyBirdGym import *
import numpy as np
import cv2

import gym

class EnvMananger():

    def __init__(self, windowMode = False):

        self.env = FlappyBirdGym(windowMode) 

        current_state = self.env.getState()
        
        self.state = np.zeros(shape=(*current_state.shape, 2))
        self.state[:, 1] = current_state
      
    
    def step(self,action):
        
        img, reward, done_flag = self.env.step(action)
       
        self.state[:, 0] = self.state[:, 1]
        self.state[:, 1] = self.env.getState()
        
        return self.state.flatten(), reward, done_flag

    def reset(self):
        self.env.reset()

     
        current_state = self.env.getState()
        self.state = np.zeros(shape=(*current_state.shape, 2))
        self.state[:, 1] = current_state
     
        return self.state.flatten()

    def getState(self):
        return self.state.flatten()

    def close(self):
        self.env.close()
    
    @property
    def observation_space_shape(self):
        return self.state.shape
    
    @property
    def action_space(self):
        return self.env.action_space
    
    
