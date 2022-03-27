import tkinter as tk
from threading import Thread
import numpy as np
import time

from FlappyBirdGym.GameLogic import *
from FlappyBirdGym.WindowMode import *
from FlappyBirdGym.Window import *


class FlappyBirdGym:

    def __init__(self, window_mode=WindowMode.NO_WINDOW):

        # Check for valid windowMode
        if window_mode not in WindowMode.available_modes:
            raise ValueError("Invalid windowMode")

        self.window_mode = window_mode

        if window_mode == WindowMode.NO_WINDOW:
            self.game_logic = GameLogic(window_mode=self.window_mode)
        else:
            self.windowThread = Thread(target=self.window_loop)
            self.windowThread.start()

            # Let the thread init gameLogic and window
            # both must exists in the thread (not in this thread)
            time.sleep(0.5)

        self.done = False

    def step(self, action):

        if action < 0 or action > 1:
            raise ValueError("Invalid action")

        if self.done:
            print("Error: Call step() on a finished game")
            return None

        done, reward = self.game_logic.nextGameStep(action)
        self.done = done

        state = self.game_logic.get_state()

        return state, reward, done

    def get_state(self):
        return self.game_logic.get_state()

    def reset(self):
        self.done = False
        self.game_logic.reset()

    def window_loop(self):
        self.root = tk.Tk()
        self.window = Window(window_mode=self.window_mode, master=self.root)
        self.game_logic = GameLogic(window_mode=self.window_mode, window=self.window)
        self.window.game_logic = self.game_logic

        self.window.mainloop()

    def close(self):
        self.game_logic.quit()

    @property
    def num_actions(self):
        return 2
