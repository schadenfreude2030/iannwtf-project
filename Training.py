import numpy as np

from Agent import *
from EnvManager import *

import matplotlib.pyplot as plt

def main():

    # Logging
    file_path = "test_logs/test" 
    train_summary_writer = tf.summary.create_file_writer(file_path)

    num_episods = 50000
    update = 100 

    env = EnvMananger()
    agent = Agent(input_dims=env.observation_space_shape,
                num_actions=2, batch_size=64)

    agent.q_net.summary()
    
    with train_summary_writer.as_default():

        for episode in range(num_episods):
            
            done_flag = False
            
            score = 0 # sum of rewards
            rewards = []

            cnt_steps = 0
            state = env.reset()
            while not done_flag and cnt_steps < 500:
                action = agent.select_action(state)
                next_state, reward, done_flag  = env.step(action)

                agent.store_experience(state, action, next_state, reward, done_flag)

                state = next_state
                agent.train_step()
                
                score += reward
          
                rewards.append(reward)
                cnt_steps += 1
            
            if agent.replayMemory.haveEnoughSamples():
                agent.strategy.reduce_epsilon()

                # save only when we are learning
                if episode % update == 0:
                    agent.update_target()
            
                # Save weights
                if episode % 100 == 0:
                    agent.q_net.save_weights(f"./saved_models/trainied_weights_epoch_{episode}", save_format="tf")
       
            tf.summary.scalar(f"Average reward", np.mean(rewards), step=episode)
            tf.summary.scalar(f"Score", score, step=episode)
            tf.summary.scalar(f"Epsilon (EpsilonGreedyStrategy)", agent.strategy.get_exploration_rate(), step=episode)
            tf.summary.scalar(f"Steps per episode", cnt_steps, step=episode)

            print(f"Episode {episode} with score {round(score, 2)} and avg reward {round(np.mean(rewards), 2)} epsilon: {agent.strategy.get_exploration_rate()}")
        
        
            # okay: 1
            # dead: -1
            # avoid: 0.01 vs 1

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")


