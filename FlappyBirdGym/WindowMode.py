from enum import Enum


class WindowMode:
    NO_WINDOW = 0
    GAME_WINDOW = 1
    GAME_WINDOW_PLOTS = 2

    available_modes = list([NO_WINDOW, GAME_WINDOW, GAME_WINDOW_PLOTS])
