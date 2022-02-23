import tkinter as tk
from FlappyBirdGym.GameWindow import *

from threading import Thread, Event

import pyautogui

import numpy as np

import time

class FlappyBirdGym:
    def __init__(self):
        
        self.e = Event()
    
        self.windowThread = Thread(target = self.windowLoop)
        self.windowThread.start() 

        self.e.wait()
        time.sleep(0.5)

        self.done = False
      

    def step(self, action):

        if self.done:
            print("Error: Call step() on a finished game")
            return None 

        done, reward = self.gameWindow.nextGameStep(action)
        self.done = done
        
        img = self.getWindowImage()

        return img, reward, done

    def getWindowImage(self):
    
        x, y = self.gameWindow.canvas.winfo_rootx(), self.gameWindow.canvas.winfo_rooty()
        w, h = self.gameWindow.canvas.winfo_width(), self.gameWindow.canvas.winfo_height()
        
        img = pyautogui.screenshot(region=(x, y, w, h))

        return np.array(img, dtype=np.float32)
    
    def reset(self):
        self.done = False
        self.gameWindow.resetGame()

    def windowLoop(self):
        root = tk.Tk()
        self.gameWindow = GameWindow(master=root)

        self.e.set()
        self.gameWindow.mainloop()
   
    def close(self):
        self.gameWindow.quit()