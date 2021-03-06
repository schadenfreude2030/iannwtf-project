import numpy as np

from Agent import *
from EnvManager import *
from FlappyBirdGym.WindowMode import *


def main():
    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    num_episods = 5000
    update = 100

    # Init gym
    env = EnvMananger(window_mode=WindowMode.NO_WINDOW)
    agent = Agent(input_dims=env.observation_space_shape,
                  num_actions=env.num_actions, batch_size=64)
    agent.q_net.summary()

    # Fill ReplayMemory until enough samples are collected
    print("Fill ReplayMemory...")
    done_flag = False
    state = env.reset()
    while not agent.replay_memory.have_enough_samples():
        action = agent.select_action(state)
        next_state, reward, done_flag = env.step(action)

        agent.store_experience(state, action, next_state, reward, done_flag)
        state = next_state
        if done_flag:
            state = env.reset()
            done_flag = False

    # Start actual training
    with train_summary_writer.as_default():

        for episode in range(num_episods):

            done_flag = False

            score = 0  # sum of rewards
            rewards = []

            cnt_steps = 0
            state = env.reset()
            while not done_flag and cnt_steps < 500:
                action = agent.select_action(state)
                next_state, reward, done_flag = env.step(action)

                agent.store_experience(state, action, next_state, reward, done_flag)

                state = next_state
                agent.train_step()

                score += reward

                rewards.append(reward)
                cnt_steps += 1

            # Reduce epsilon after each episode
            agent.strategy.reduce_epsilon()

            # Update target network
            if episode % update == 0:
                agent.update_target()

            # Save weights
            if episode % 10 == 0:
                agent.q_net.save_weights(f"./saved_models/trainied_weights_epoch_{episode}", save_format="tf")

            tf.summary.scalar(f"Average reward", np.mean(rewards), step=episode)
            tf.summary.scalar(f"Score", score, step=episode)
            tf.summary.scalar(f"Epsilon (EpsilonGreedyStrategy)", agent.strategy.get_exploration_rate(), step=episode)
            tf.summary.scalar(f"Steps per episode", cnt_steps, step=episode)

            print(f"  Episode: {episode}")
            print(f"  Epsilon: {round(agent.strategy.get_exploration_rate(), 2)}")
            print(f"    Score: {round(score, 2)}")
            print(f"Avg Score: {round(np.mean(rewards), 2)}")
            print(f"    Steps: {cnt_steps}")
            print("------------------------")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
