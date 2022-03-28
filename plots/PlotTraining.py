from re import A
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append("../")

def main():
    
    df_avg_reward = pd.read_csv('../TensorBoard_csv_files/run-test-tag-Average reward.csv', sep=',')
    avg_reward = df_avg_reward["Value"]
  
    df_epsilon = pd.read_csv('../TensorBoard_csv_files/run-test-tag-Epsilon (EpsilonGreedyStrategy).csv', sep=',')
    epsilon = df_epsilon["Value"]

    df_score = pd.read_csv('../TensorBoard_csv_files/run-test-tag-Score.csv', sep=',')
    score = df_score["Value"]

    df_stepsPerEpisode = pd.read_csv('../TensorBoard_csv_files/run-test-tag-Steps per episode.csv', sep=',')
    stepsPerEpisode = df_stepsPerEpisode["Value"]

    x = np.arange(len(epsilon))

    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].plot(x, avg_reward)
    ax[0].set_title("Average reward per episode")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Average reward")
    ax[0].grid(True)

    ax[1].plot(x, epsilon)
    ax[1].set_title("Epsilon Greedy Strategy")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Epsilon")
    ax[1].grid(True)

    ax[2].plot(x, score)
    ax[2].set_title("Score per episode")
    ax[2].set_xlabel("Episode")
    ax[2].set_ylabel("Score")
    ax[2].grid(True)

    ax[3].plot(x, stepsPerEpisode)
    ax[3].set_title("Steps per episode")
    ax[3].set_xlabel("Episode")
    ax[3].set_ylabel("Steps")
    ax[3].grid(True)

    #plt.tight_layout()
    fig.set_size_inches(w=17.5, h=4)
    plt.savefig("../media/trainingPlot_noSmoothing.png")
    plt.show()

    



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")