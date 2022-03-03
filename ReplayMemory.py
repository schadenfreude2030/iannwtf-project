import numpy as np

class ReplayMemory():
    def __init__(self, capacity, input_dims):
        self.capacity = capacity
        self.idx = 0
        self.idx_wasOverflown = False

        # experience = state, action, next_state, reward
        self.states = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.next_states = np.zeros((self.capacity, *input_dims), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)

        self.done_flags = np.zeros(self.capacity, dtype=np.int32)

    def store_experience(self, state, action, next_state, reward, done_flag):
        
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward
        
        self.done_flags[self.idx] = 1 - int(done_flag)

        self.idx += 1
        if self.capacity <= self.idx:
            self.idx_wasOverflown = True 
            self.idx = 0

    def sample_batch(self, batch_size):

        if self.idx_wasOverflown:
            max_mem = self.capacity
        else:
            max_mem = self.idx
        
        # Thompson sampling
        rewards = self.rewards[:max_mem] 
        # Normalize between [0,1] 
        rewards_z = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
        probs = rewards_z/np.sum(rewards_z) # sum up each value must be 1
   
        # A value shall not be sampled mutiple times within a batch
        sampled_idxs = np.random.choice(max_mem, batch_size, replace=False, p=probs)

        states = self.states[sampled_idxs]
        actions = self.actions[sampled_idxs]
        next_state = self.next_states[sampled_idxs]
        rewards = self.rewards[sampled_idxs]
        
        done_flag = self.done_flags[sampled_idxs]

        return states, actions, next_state, rewards, done_flag
    
    def haveEnoughSamples(self):
        return self.idx_wasOverflown or 150000 < self.idx 
