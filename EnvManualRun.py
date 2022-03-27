from FlappyBirdGym.FlappyBirdGym import *
import matplotlib.pyplot as plt

import numpy as np


def main():
    gym = FlappyBirdGym(window_mode=False)

    while True:
        action = input("Enter: '1' (jump) , '' (nothing): ")
        if action == "":
            action = 0
        state, reward, done = gym.step(int(action))

        print("Reward: ", reward, "Done: ", done)

        if done:
            gym.reset()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
