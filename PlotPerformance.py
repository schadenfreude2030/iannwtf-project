import numpy as np
import matplotlib.pyplot as plt

from DDQN import *
from EnvManager import *
from FlappyBirdGym.FlappyBirdGym import *
from FlappyBirdGym.WindowMode import *

def main():
    env = EnvMananger(window_mode=WindowMode.NO_WINDOW)

    # Load model
    q_net = DDDQN(num_actions=env.num_actions)

    q_net.build((1, *env.observation_space_shape))  # need a batch size
    q_net.load_weights("./saved_models/trainied_weights_epoch_810")

    q_net.summary()

    state = env.get_state()

    collectedRewards = []
    NUM_STEPS = 250
    for i in range(NUM_STEPS):
        
        # Add batch dim
        state = np.expand_dims(state, axis=0)

        # Predict best action
        target = q_net(state)

        target = target[0]  # Remove batch dim
        best_action = np.argmax(target)

        # Execute best action
        state, reward, done = env.step(best_action)
        collectedRewards.append(reward)
    
        if done:
            env.reset()

    x = np.arange(NUM_STEPS)

    plt.plot(x, collectedRewards, label="Reward")
   
    plt.title("Obtained rewards")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.ylim(0, 1)

    mean_collected_rewards = np.mean(collectedRewards)

    plt.axhline(mean_collected_rewards, color='r', linestyle="--", label="Mean")
    plt.legend(loc='lower right')
    
    plt.savefig("./media/performancePlot.png")



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")