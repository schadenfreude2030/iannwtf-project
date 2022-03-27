import numpy as np


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.epsilon = start
        self.end = end

        self.decay = decay

    def get_exploration_rate(self):
        return self.epsilon

    def reduce_epsilon(self):
        # if self.epsilon - self.decay > self.end:
        #     self.epsilon -= self.decay

        new_epsilon = self.epsilon * self.decay
        if self.end < new_epsilon:
            self.epsilon = new_epsilon

        # expo decay -> better curve
