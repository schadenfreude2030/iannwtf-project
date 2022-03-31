import tkinter as tk
from threading import Thread
import numpy as np
import time

from FlappyBirdGym.GameLogic import *
from FlappyBirdGym.WindowMode import *
from FlappyBirdGym.Window import *


class FlappyBirdGym:

    def __init__(self, window_mode=WindowMode.NO_WINDOW):

        """Init the FlappyBirdGym. It encapsulates the window and holding its 
        window thread.

        Keyword arguments:
        window_mode -- 0 = no window, 1 = only game window, 2 = game window with plots
        """

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

    def step(self, action: int):
        """Interacting with the environment by doing an action

        Keyword arguments:
        action -- action of the bird: 0 = no jump, 1 = jump

        Return:
        next game state, reward, done (does the action end the game?)
        """

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
        """
        Return:
        current game state 
        """
        return self.game_logic.get_state()

    def reset(self):
        """
        Return:
        Reset the current game
        """
        self.done = False
        self.game_logic.reset()

    def window_loop(self):
        """
        Window loop of the game window. 
        Calling this method will block the calling thread.
        """
        self.root = tk.Tk()
        self.window = Window(window_mode=self.window_mode, master=self.root)
        self.game_logic = GameLogic(window_mode=self.window_mode, window=self.window)
        self.window.game_logic = self.game_logic

        self.window.mainloop()

    def close(self):
        """
        Closes the game window
        """
        self.game_logic.quit()

    @property
    def num_actions(self):
        """
        Return the number of available actions.
        There are 2 actions (no jump and jump) available.
        """
        return 2
