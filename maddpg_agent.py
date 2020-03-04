
import numpy as np
import random

from collections import namedtuple, deque
from models import DDPG_Actor,DDPG_Critic
from utils import ExperienceReplay,PrioritizedExperienceReplay,NormalNoiseStrategy
# from prioritized_memory import Memory

import torch
import torch.nn.functional as F
import torch.optim as optim

class MADDPG_Agent():
    def __init__(self, state_size, action_size, num_agents, \
                 gamma=0.99, tau=1e-3, lr_actor=1e-3, lr_critic=1e-2, \
                 buffer_size = 1e5, buffer_type = 'replay', policy_update = 1, \
                 noise_init = 1.0, noise_decay=0.9995, min_noise=0.1):
        # General info
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.t_step = 0
        self.gamma = gamma
        # Actor Networks -- Policy-based
        self.actors = [DDPG_Actor(state_size, action_size, hidden_dims=(128,128)) for i in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        # targets
        self.target_actors = [DDPG_Actor(state_size, action_size, hidden_dims=(128,128)) for i in range(num_agents)]
        [self.hard_update(self.actors[i],self.target_actors[i]) for i in range(num_agents)]
        # Critic Network -- Value-based --> in this approach we will use one common network for all the actors
        self.critic = DDPG_Critic(state_size, action_size, hidden_dims=(128,128))
        self.target_critic = DDPG_Critic(state_size, action_size, hidden_dims=(128,128))
        self.hard_update(self.critic,self.target_critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        # How to update networks
        self.tau = tau
        self.policy_update = policy_update
        # Replay memory
        self.buffer_type = buffer_type
        self.memory = ExperienceReplay(action_size, int(buffer_size)) #ExperienceReplay
        self.per = PrioritizedExperienceReplay(capacity=int(buffer_size),alpha=0.6,beta=0.9,error_offset=0.001)
        # NormalNoiseStrategy
        self.normal_noise = NormalNoiseStrategy(noise_init=noise_init,\
                                                noise_decay=noise_decay,\
                                                min_noise_ratio = min_noise)

    def select_action(self, state):
        actions = []
        for i in range(self.num_agents):
            actions.append(self.normal_noise.select_action(self.actors[i], state[i]))
        return np.array(actions)

    def select_action_evaluation(self,state):
        actions = []
        for i in range(self.num_agents):
            actions.append(self.actors[i](state[i]).cpu().detach().data.numpy().squeeze())
        return np.array(actions)

    def _critic_error(self, state, action, reward, next_state, done):
        states = torch.Tensor(state).view(-1,self.num_agents*self.state_size) # batch X 2*24
        next_states = torch.Tensor(next_state).view(-1,self.num_agents*self.state_size) # batch X 2*24
        actions = torch.Tensor(action).view(-1, self.num_agents*self.action_size) # batch X 2*2
        rewards = torch.Tensor(reward).view(-1, self.num_agents*1)
        dones = torch.Tensor(done.astype(int)).view(-1, self.num_agents*1)

        with torch.no_grad():
            # 1.1. Calculate Target
            target_actions = []
            for i in range(self.num_agents):
                target_actions.append(self.target_actors[i](next_states[:,self.state_size*i:self.state_size*(i+1)]))
            target_actions = torch.stack(target_actions) # shape: 2(num_agents) x batch x 2(num_actions)
            target_actions = target_actions.permute(1,0,2) # transform from 2 X batch_size X 2 --> batch_size X 2 X 2
            target_actions = target_actions.contiguous().view(-1, self.num_agents*self.action_size) # batch_size X 2*2
            q_target_next = self.target_critic(next_states, target_actions)

            q_target = rewards + (self.gamma*q_target_next*(1-dones)) # we get batch_size X 2 (one q target for each agent --> we have rewards and dones for each agent)
            # 1.2. Expected
            q_expected = self.critic(states,actions)
            # 1.3. Compute loss
            td_error = q_expected - q_target.detach()
        return td_error.mean().detach().numpy()

    def step(self, state, action, reward, next_state, done, batch_size = 64):
        self.t_step += 1 #increment number of visits
        # transform to np.array with proper shapes
        reward = np.asarray(reward)[:,np.newaxis]
        done = np.asarray(done)[:,np.newaxis]
        # add experiences to buffer(PER | Replay) and learn in case of having enough samples
        if self.buffer_type == 'prioritized':
            for i in range(self.num_agents):
                error = self._critic_error(state,action,reward,next_state,done)
                self.per.add(error, (state, action, reward, next_state, done))
            # train if enough samples
            if self.t_step > batch_size:
                experiences, mini_batch, idxs, is_weights = self.per.sample(batch_size)
                self.learn(experiences,batch_size,idxs,is_weights)
        else: #replaybuffer
            self.memory.add(state, action, reward, next_state, done)
            # train if enough samples
            if len(self.memory) > batch_size:
                experiences = self.memory.sample(batch_size)
                c_loss, a_loss = self.learn(experiences,batch_size)
            else:
                c_loss, a_loss = torch.Tensor([0]),(torch.Tensor([0]),torch.Tensor([0]))
        return c_loss, a_loss

    def _update_critic_network(self,experiences,batch_size,idxs,is_weights):
        states, actions, rewards, next_states, dones = experiences
        # s,s' --> 64x2x24
        # a --> 64x2x2
        # r,w --> 64x2x1

        # transform to proper shape for the network --> batch_size X expected value
        states = states.view(-1,self.num_agents*self.state_size) # batch X 2*24
        next_states = next_states.view(-1,self.num_agents*self.state_size) # batch X 2*24
        actions = actions.view(-1, self.num_agents*self.action_size) # batch X 2*2
        rewards = rewards.view(-1, self.num_agents*1)
        dones = dones.view(-1, self.num_agents*1)

        # 1.1. Calculate Target
        target_actions = []
        for i in range(self.num_agents):
            target_actions.append(self.target_actors[i](next_states[:,self.state_size*i:self.state_size*(i+1)]))
        target_actions = torch.stack(target_actions) # shape: 2(num_agents) x batch x 2(num_actions)
        # transform to proper shape
        target_actions = target_actions.permute(1,0,2) # transform from 2 X batch_size X 2 --> batch_size X 2 X 2
        target_actions = target_actions.contiguous().view(-1, self.num_agents*self.action_size) # batch_size X 2*2

        q_target_next = self.target_critic(next_states, target_actions)

        q_target = rewards + (self.gamma*q_target_next*(1-dones)) # we get batch_size X 2 (one q target for each agent --> we have rewards and dones for each agent)
        # 1.2. Expected
        q_expected = self.critic(states,actions)
        # 1.3. Compute loss
        td_error = q_expected - q_target.detach()

        if self.buffer_type == 'prioritized':
            # PER --> update priority
            with torch.no_grad():
                error = td_error.detach().numpy()
                for i in range(batch_size):
                    idx = idxs[i]
                    self.per.update(idx, error[i])
            value_loss = (torch.FloatTensor(is_weights) * td_error.pow(2).mul(0.5)).mean()
        else:
            value_loss = td_error.pow(2).mul(0.5).mean()
            # value_loss = F.mse_loss(q_expected,q_target)
        # 1.4. Update Critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        
        return value_loss

    def _update_actor_networks(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # transform to proper shape for the network --> batch_size X expected value
        states = states.view(-1,self.num_agents*self.state_size) # batch X 2*24
        next_states = next_states.view(-1,self.num_agents*self.state_size) # batch X 2*24
        actions = actions.view(-1, self.num_agents*self.action_size) # batch X 2*2
        rewards = rewards.view(-1, self.num_agents*1)
        dones = dones.view(-1, self.num_agents*1)

        policy_losses = []
        for ID_actor in range(self.num_agents):
            # load network and optimizer
            optimizer = self.actor_optimizers[ID_actor]
            actor = self.actors[ID_actor]

            q_input_actions = []
            for i in range(self.num_agents):
                q_input_actions.append(actor(states[:,self.state_size*i:self.state_size*(i+1)])) #only states of the current agent
            q_input_actions = torch.stack(q_input_actions)
            # transform to proper shape
            q_input_actions = q_input_actions.permute(1,0,2) # transform from 2 X batch_size X 2 --> batch_size X 2 X 2
            q_input_actions = q_input_actions.contiguous().view(-1, self.num_agents*self.action_size) # batch_size X 2*2

            max_val = self.critic(states,q_input_actions)
            policy_loss = -max_val.mean() # add minus because its gradient ascent
            policy_losses.append(policy_loss)
            
            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[ID_actor].parameters(), 1)
            optimizer.step()

            # save new network and optimizer state
            self.actor_optimizers[ID_actor] = optimizer
            self.actors[ID_actor] = actor
        
        return policy_losses[0],policy_losses[1]

    def learn(self,experiences,batch_size,idxs=0,is_weights=0):
        # *** 1. UPDATE Online Critic Network ***
        critic_loss = self._update_critic_network(experiences,batch_size,idxs,is_weights)
        if self.t_step % self.policy_update == 0:
            # *** 2. UPDATE Online Actor Networks ***
            actor_loss = self._update_actor_networks(experiences)
            # *** 3. UPDATE TARGET/Offline networks ***
            for i in range(self.num_agents):
                self.soft_update(self.actors[i],self.target_actors[i],self.tau)
            self.soft_update(self.critic,self.target_critic,self.tau)
        return critic_loss, actor_loss

    def hard_update(self, local_model, target_model):
        """Hard update model parameters. Copy the values of local network into the target.
        θ_target = θ_local

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
