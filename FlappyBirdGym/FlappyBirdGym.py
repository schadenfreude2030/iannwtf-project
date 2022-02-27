import tkinter as tk
from FlappyBirdGym.GameLogic import *

from threading import Thread, Event

import numpy as np

import time

class FlappyBirdGym:
    def __init__(self, windowMode = False):
        
        if windowMode:
            self.e = Event()
    
            self.windowThread = Thread(target = self.windowLoop)
            self.windowThread.start() 

            self.e.wait()
            time.sleep(0.5)

        else:
            self.gameLogic = GameLogic(windowMode=False)

        self.done = False
      

    def step(self, action):

        if self.done:
            print("Error: Call step() on a finished game")
            return None 

        done, reward = self.gameLogic.nextGameStep(action)
        self.done = done
        
        state = self.gameLogic.getState()
      
        return state, reward, done
    
    def getState(self):
        return self.gameLogic.getState()

    
    def reset(self):
        self.done = False
        self.gameLogic.reset()

    def windowLoop(self):
        root = tk.Tk()
        self.gameLogic = GameLogic(windowMode=True, master=root)

        self.e.set()
        self.gameLogic.mainloop()
   
    def close(self):
        self.gameLogic.quit()