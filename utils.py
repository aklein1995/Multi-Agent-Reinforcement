import torch
import random
import numpy as np
from collections import namedtuple, deque

class ExperienceReplay:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size):
        """Initialize a ExperienceReplay object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        """
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

"""
Prioritized Experience Replay code used from:
https://github.com/rlcode/per
"""
class PrioritizedExperienceReplay:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity, alpha = 0.6, beta = 0.4, beta_increment_per_sampling = 0, error_offset = 0.01):
        """
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        beta: float
                To what degree to use importance weights
                (0 - no corrections, 1 - full correction)
        """
        self.a = alpha
        self.e = error_offset
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def process_samples(self,batch):
        mini_batch = np.array(batch).transpose()
        try:
            s = torch.from_numpy(np.vstack(mini_batch[0])).float()
            a = torch.from_numpy(np.vstack(mini_batch[1])).float()
            r = torch.from_numpy(np.vstack(mini_batch[2])).float()
            ns = torch.from_numpy(np.vstack(mini_batch[3])).float()
            d = torch.from_numpy(np.vstack(mini_batch[4]).astype(np.uint8)).float()
        except ValueError:
            print('batch len(exp 256): ',len(batch))
            print('batch[0] len(exp 5): ',len(batch[0]))
            print('array conversion --> shape: ',np.array(batch).shape)
        return (s,a,r,ns,d)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            while True:
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                if not isinstance(data, int):
                    break
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        """
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        """
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        experiences = self.process_samples(batch)

        return experiences,batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class NormalNoiseStrategy():
    def __init__(self, bounds=(-1,1), noise_init = 1.0, noise_decay = 0.9995, exploration_noise_ratio=0.1):
        self.low, self.high = bounds
        self.exploration_noise_ratio = exploration_noise_ratio
        self.min_noise = self.exploration_noise_ratio * self.high
        self.noise_decay = noise_decay
        self.noise_scale = noise_init
        self.ratio_noise_injected = 0

    def select_action(self, model, state, max_exploration=False):
        """
            Get actions and add gaussian normal noise clipping output values in
            valid ranges
        """
        if max_exploration:
            self.noise_scale = self.high
        # elif :
        #     self.noise_scale = self.exploration_noise_ratio * self.high
        else:
            self.noise_scale = max(self.noise_scale*self.noise_decay, self.min_noise)
            
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        noise = np.random.normal(loc=0, scale=self.noise_scale, size=1)
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)

        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))

        return action
