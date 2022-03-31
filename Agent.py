import numpy as np

from ReplayMemory import *
from DDQN import *
from EpsilonGreedyStrategy import *


class Agent:
    def __init__(self, num_actions: int, batch_size: int, input_dims: tuple):
        """Init the Agent by creating the EpsilonGreedyStrategy, ReplayMemory
        q-network and target network. 

        Keyword arguments:
        num_actions -- Number of possible actions which can be taken in the gym.
        batch_size -- batch size, number of samples which are sampled from the replay memory during each train step
        input_dims -- dimension of a both game states (previous AND current game step concatenated)
        """

        self.gamma = 0.99

        self.num_actions = num_actions

        self.batch_size = batch_size

        self.strategy = EpsilonGreedyStrategy(start=1.0, end=0.05, decay=0.99)
        self.replay_memory = ReplayMemory(capacity=500000, input_dims=input_dims)

        self.q_net = DDDQN(num_actions)
        self.q_net.build((self.batch_size, *input_dims))

        self.target_net = DDDQN(num_actions)
        self.target_net.build((self.batch_size, *input_dims))
        self.update_target()

    def select_action(self, state: np.array):
        """Based on the game state (see parameter) a action will be choosen.
        exploration vs exploitation by epsilon greedy strategy

        Keyword arguments:
        state : current state (input of the IANN), previous and current game step concatenated

        Return:
        choosen action: 0 = no jump, 1 = jump
        """

        # Exploration
        if np.random.random() < self.strategy.get_exploration_rate():
            return np.random.randint(0, self.num_actions)
        # Exploitation
        else:
            # Add batch dim
            state = np.expand_dims(state, axis=0)

            # Select best action
            actions = self.q_net(state)
            return np.argmax(actions)

    def store_experience(self, state, action, next_state, reward, done):
        """Store a experience in the ReplayMemory.
        A experience consists of a state, an action, a next_state and a reward.

        Keyword arguments:
        state -- game state 
        action -- action taken in state
        next_state -- the new/next game state: in state do action -> next_state
        reward -- reward received
        done_flag -- does the taken action end the game?
        """

        self.replay_memory.store_experience(state, action, next_state, reward, done)

    def update_target(self):
        """The target network's weights are set to the q-network's weights. 
        """
        # Polyak averaging would be nice
        self.target_net.set_weights(self.q_net.get_weights())

    def train_step(self):

        """
        A random batch is sampled from the ReplayMemory. 
        Thereafter, the q-network is trained.
        Note that, enough samples in ReplayMemory must be in the ReplayMemory. 
        Otherwise, there will be no training of the network.
        """

        # If not enough samples in ReplayMemory -> return
        if not self.replay_memory.have_enough_samples():
            return

        # Sample a random batch
        states, actions, next_state, rewards, dones = \
            self.replay_memory.sample_batch(self.batch_size)

        # ---------

        # Inspired from https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a

        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_state)
 
        max_action = np.argmax(self.q_net.predict(next_state), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)  

        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones
        self.q_net.train_step(states, q_target)

        # ------