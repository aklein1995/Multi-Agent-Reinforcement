#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alain
"""

from maddpg_agent import MADDPG_Agent
from unityagents import UnityEnvironment

import numpy as np
import pickle
import torch
from collections import deque

###############################################################################
# hyperparams
GOAL = 0.5
EPISODES = 10000
GAMMA = .99
TAU = 1e-2
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
BUFFER_SIZE = int(1e5)
BUFFER_TYPE = 'replay'#'prioritized'
BATCH_SIZE = 128
POLICY_UPDATE = 1 # for normal updates =1
SEED = 0
# other required params
path = 'Tennis_Linux/Tennis.x86_64'
algorithm = 'MADDPG'
mode = 'train'#'evaluation'
results_filename = 'scores/scores_' + algorithm + '_' +mode
###############################################################################

def plotResults(scores,checkpoint=0,compacted = True):
    import matplotlib.pyplot as plt
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if compacted:
        best_score = np.max(scores) + 1
        ax.set_ylim([0,best_score])
        plt.axvline(x=checkpoint,ymax = GOAL/best_score,linewidth = 1, color = 'red', linestyle = ':')
        plt.title('Avg score obtained in 100 consecutive episodes')
    else:
        plt.title('Scores obtained in each episode')
    plt.plot(np.arange(len(scores)), scores)
    plt.axhline(y=GOAL, linewidth = 1, color = 'black', linestyle = '--')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def defineEnvironment(path,verbose=False):
    # set the path to match the location of the Unity environment
    env = UnityEnvironment(file_name=path, worker_id= np.random.randint(0,int(10e6)))
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    if verbose:
        print('Number of agents:', len(env_info.agents))
        print('Number of actions:', action_size)
        print('States have length:', state_size)
    return env, brain_name, state_size, action_size, len(env_info.agents)

def playRandomAgent(env,brain_name,action_size=4,num_agents=1):
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

def train_agent(agent,env,brain_name,n_episodes=300, batch_size = BATCH_SIZE):
    num_visits = []
    scores, scores1, scores2 = [], [], []
    scores_window = deque(maxlen=100)
    print_stats_every = 10
    print_scores_every = 100
    stop_criteria = GOAL #+0.5 (over 100 consecutive episodes)
    aux = True
    checkpoint, checkpoint_best = 0,0
    best_weights = [0,0,0]
    best_score = 1.5
    max_episodes = n_episodes
    e = -1
    # ------------------- begin training ------------------- #
    while True:
        # --- New Episode --- #
        e += 1
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get the current state
        state = env_info.vector_observations
        score = np.zeros(2)
        visits = 0
        # --- Generate trajectories --- #
        while True:
            visits += 1
            # get value of the 4 continuous actions
            action = agent.select_action(state)
            # get reward & next_states
            env_info = env.step(action)[brain_name]           # send all actions to the environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            # record trajectory
            agent.step(state,action,reward,next_state,done, batch_size)
            # update score
            score += np.array(env_info.rewards)
            #check if finished
            if np.any(done):
                print('# visits: ',visits)
                break
            else:
                state = next_state

        # Update monitorization variables & params for next Episode
        num_visits.append(visits)
        scores1.append(score[0])
        scores2.append(score[1])
        scores.append(np.max(score))
        scores_window.append(np.max(score))
       
        if e % print_stats_every == 0:
            print('Episode {}/{}\tLast 10 episodes: {:.5f}\tAvg Score: {:.5f}'.format(e,n_episodes,np.mean(scores[-10:]),np.mean(scores_window)))
        
        if e % print_scores_every == 0:
            plotResults(scores,0,compacted = False)
            
        if np.mean(scores_window) >= stop_criteria and aux:
            print('Environment solved in {} episodes'.format(e))
            aux = False
            checkpoint = e
            max_episodes = e + 500 # UPDATE THE NUMBER OF MAX_EPISODES
            torch.save(agent.actors[0].state_dict(), 'weights/actor' + str(1) + '_model_weights_checkpoint.pth')
            torch.save(agent.actors[1].state_dict(), 'weights/actor' + str(2) + '_model_weights_checkpoint.pth')
            torch.save(agent.critic.state_dict(), 'weights/critic_model_weights_checkpoint.pth')
            
            best_weights[0] = agent.actors[0].state_dict()
            best_weights[1] = agent.actors[1].state_dict()
            best_weights[2] = agent.critic.state_dict()
       
        if np.mean(scores[-10:]) > best_score: # max score=2.5 --> if last10 ep avg more than 2 --> save weights
            best_score = np.mean(scores[-10:])
            checkpoint_best = e
            best_weights[0] = agent.actors[0].state_dict()
            best_weights[1] = agent.actors[1].state_dict()
            best_weights[2] = agent.critic.state_dict()
        
        # STOP CRITERIA
        if e >= max_episodes or np.mean(scores_window) > 2.0:
            break
        
    # save the model weights
    if checkpoint > 0:
        print('Environment solved in {} episodes'.format(checkpoint))
    print('best_score: {:.2f} obtained in {} episode'.format(best_score,checkpoint_best))
    torch.save(best_weights[0], 'weights/actor' + str(1) + '_model_weights.pth')
    torch.save(best_weights[1], 'weights/actor' + str(2) + '_model_weights.pth')
    torch.save(best_weights[2], 'weights/critic_model_weights.pth')

    return scores,scores1,scores2,checkpoint,checkpoint_best

def evaluate_agent(agent,env,brain_name,n_episodes=1):
    scores = []
    # ------------------- begin training ------------------- #
    for e in range(1,n_episodes+1):
        # --- New Episode --- #
        # reset the environment
        env_info = env.reset(train_mode=False)[brain_name]
        # get the current state
        state = env_info.vector_observations
        score = np.zeros(2)
        # --- Visits --- #
        while True:
            # Agent selects an action
            action = agent.select_action_evaluation(state)
            # get reward & next_states
            env_info = env.step(action)[brain_name]           # send all actions to the environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            # Update monitorization variables & params for next visit
            score += np.array(env_info.rewards)
            if np.any(done):
                break
            else:
                state = next_state
        # Update monitorization variables & params for next Episode
        print('Episode/Test {} throws an avg of p1:{:.3f} / p2:{:.3f}'.format(e,score[0],score[1]))
        scores.append(np.max(score))
    return scores

if __name__ == "__main__":
    # set environment and get state & action size
    env, brain_name, state_size,action_size, num_agents = defineEnvironment(path,verbose=True)

    # define agent
    agent = MADDPG_Agent(state_size,action_size,num_agents,SEED, \
                         GAMMA, TAU, LR_ACTOR, LR_CRITIC, BUFFER_SIZE, \
                         BUFFER_TYPE, POLICY_UPDATE)

    if mode == 'train':
        scores,s1,s2,checkpoint,checkpoint_best = train_agent(agent,env,brain_name,n_episodes=EPISODES, batch_size = BATCH_SIZE)
        # export data
        with open(results_filename,'wb') as f:
            pickle.dump([scores,s1,s2,checkpoint,checkpoint_best],f)
    elif mode == 'evaluation':
        w1 = 'weights/actor1_model_weights_checkpoint.pth'
        w2 = 'weights/actor2_model_weights_checkpoint.pth'
        agent.actors[0].load_state_dict(torch.load(w1))
        agent.actors[0].eval()
        agent.actors[1].load_state_dict(torch.load(w2))
        agent.actors[1].eval()
        scores = evaluate_agent(agent,env,brain_name,n_episodes=EPISODES)
        checkpoint = None
    # close env
    env.close()
