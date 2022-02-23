#import gym
from FlappyBirdGym.FlappyBirdGym import *
import numpy as np
import cv2

class EnvMananger():

    def __init__(self):
        self.env = FlappyBirdGym() #gym.make('LunarLander-v2')#.unwrapped
        #self.env.reset()
     
        current_img = self.getImage()#env.getWindowImage()
        self.state = np.zeros(shape=(*current_img.shape, 2))
        self.state[:, :, 1] = current_img
      
    
    def step(self,action):
        
        img, reward, done_flag = self.env.step(action)

        
        self.state[:, :, 0] = self.state[:, :, 1]
        self.state[:, :, 1] = self.getImage()
     
        return self.state, reward, done_flag

    def reset(self):
        self.env.reset()

     
        current_img = self.getImage()#env.getWindowImage()
        self.state = np.zeros(shape=(*current_img.shape, 2))
        self.state[:, :, 1] = current_img
     
        return self.state

    def getState(self):
        return self.state

    def getImage(self):
        # Get image
        img = self.env.getWindowImage()#render(mode="rgb_array")

        # Shrink
        img = cv2.resize(img, (225, 150)) 

        img = img[:, :100, :]

        # Convert RGB -> Grey color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
        img = (img/128)-1        
        
        return img

    def close(self):
        self.env.close()
    
    @property
    def observation_space_shape(self):
        return self.state.shape
    
    @property
    def action_space(self):
        return self.env.action_space
    
    
