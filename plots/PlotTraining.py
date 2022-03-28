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
    plt.plot(x, stepsPerEpisode)
    plt.show()

    #plt.savefig("../media/performancePlot.png")



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")