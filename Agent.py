import numpy as np

from ReplayMemory import *
from DDQN import *
from EpsilonGreedyStrategy import *

class Agent():
    def __init__(self, num_actions, batch_size, input_dims):

        self.gamma = 0.99

        self.num_actions = num_actions

        self.batch_size = batch_size

        self.strategy = EpsilonGreedyStrategy(start=1.0, end=0.05, decay=0.99)
        self.replayMemory = ReplayMemory(capacity=75000, input_dims=input_dims)

        self.q_net = DDDQN(num_actions)
        self.q_net.build((self.batch_size ,*input_dims))

        self.target_net = DDDQN(num_actions)
        self.target_net.build((self.batch_size ,*input_dims))
        self.update_target()

    def select_action(self, state):
        
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
        self.replayMemory.store_experience(state, action, next_state, reward, done)

    def update_target(self):
        # Polyak averaging
          self.target_net.set_weights(self.q_net.get_weights()) 
    
    def train_step(self):
        if not self.replayMemory.haveEnoughSamples():
            return

        states, actions, next_state, rewards, dones = \
                self.replayMemory.sample_batch(self.batch_size)

        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_state)
        max_action = np.argmax(self.q_net.predict(next_state), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)  #optional  
        
        # print(rewards.shape)
        # print(next_state_val.shape)
        # print(batch_index.shape)
        # print(max_action.shape)
        # print(dones.shape)

        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones
        self.q_net.train_step(states, q_target)
     
        #self.strategy.reduce_epsilon() # each episode
        # erst wenn buffer "fast voll"
        # erst samples holen bevor trainieren
        # thompson sampling


       


    