from FlappyBirdGym import *
import matplotlib.pyplot as plt


def main():

    gym = FlappyBirdGym()
    
    while True:
        action = input("Enter: '1' (jump) , '' (nothing): ")
        img, reward, done = gym.step(action)

        print("Reward: ", reward , "Done: ", done)
        plt.imshow(img)
        plt.savefig("test.png")

        if done:
            gym.reset()
      

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")