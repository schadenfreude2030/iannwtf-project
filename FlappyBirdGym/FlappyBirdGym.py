import tkinter as tk
from threading import Thread
import numpy as np
import time

from FlappyBirdGym.GameLogic import *
from FlappyBirdGym.WindowMode import * 
from FlappyBirdGym.Window import *

class FlappyBirdGym:


    def __init__(self, windowMode = WindowMode.NO_WINDOW):
        
        # Check for valid windowMode
        if windowMode not in WindowMode.availableModes:
            raise ValueError("Invalid windowMode")

        self.windowMode = windowMode

        if windowMode == WindowMode.NO_WINDOW:
            self.gameLogic = GameLogic(windowMode=self.windowMode)
        else:
            self.windowThread = Thread(target = self.windowLoop)
            self.windowThread.start() 

            # Let the thread init gameLogic and window
            # both must exists in the thread (not in this thread)
            time.sleep(0.5)

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
        self.root = tk.Tk()
        self.window = Window(windowMode=self.windowMode, master=self.root)
        self.gameLogic = GameLogic(windowMode=self.windowMode, window=self.window)
        self.window.gameLogic = self.gameLogic
       
        self.window.mainloop()
   
    def close(self):
        self.gameLogic.quit()
    
    @property
    def action_space(self):
        return 2