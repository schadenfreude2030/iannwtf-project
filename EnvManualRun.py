from FlappyBirdGym.FlappyBirdGym import *
import matplotlib.pyplot as plt

import numpy as np

def main():

    # x = np.arange(0,450)
    # y = gaussianNormal(x, mu=250, sigma=100)*200
    # plt.plot(x,y)
    # plt.show()
    # exit()

    gym = FlappyBirdGym()
    
    while True:
        action = input("Enter: '1' (jump) , '' (nothing): ")
        if action == "":
            action = 0
        img, reward, done = gym.step(int(action))

        print("Reward: ", reward , "Done: ", done)
        #plt.imshow(img)
        #plt.savefig("test.png")

        if done:
            gym.reset()
      
def gaussianNormal(x, mu=0, sigma=1):
        return (1/ (sigma*np.sqrt(2*np.pi)) )* np.exp((-1/2)* ((x-mu)/sigma)**2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")