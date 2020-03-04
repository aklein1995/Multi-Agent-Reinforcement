#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alain
"""

from maddpg_agent import MADDPG_Agent
from unityagents import UnityEnvironment

import numpy as np
import torch
from collections import deque
from plot_utils import plotResults,plotResults_overlap,scoresEvery100episodes

import os
import pickle
###############################################################################
# hyperparams
GOAL = 0.5
EPISODES = 3000
GAMMA = .99
TAU = 1e-2
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
BUFFER_SIZE = int(1e5)
BUFFER_TYPE = 'replay'#'prioritized'
BATCH_SIZE = 256
POLICY_UPDATE = 1
#NOISE RELATED PARAMS
NOISE_INIT = 1.0
NOISE_DECAY = 0.9995
NOISE_MIN = 0.1
# other required params
env_path = 'Tennis_Linux/Tennis.x86_64'
mode = 'evaluation'#'evaluation'
###############################################################################

def generateTestFolder():
    folders = os.listdir('results')
    max_v = 0
    for f in folders:
        count = int(f.split('_')[1])
        if count > max_v:
            max_v = count
    new_v = max_v + 1
    result_folder_path = 'results/test_' + str(new_v)
    os.mkdir(result_folder_path)
    return result_folder_path

def defineEnvironment(path,verbose=False):
    # set the path to match the location of the Unity environment
    print(path)
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
    max_episodes = n_episodes
    print_stats_every = 10
    print_scores_every = 100
    stop_criteria = GOAL
    aux = True
    checkpoint, checkpoint_best = 0,0
    weights_goal_checkpoint = [0,0,0]
    weights_best_checkpoint = [0,0,0]
    best_score = 2.0
    std_out = '## STD_OUT - TRAINING LOG'
    critic_losses, actor1_losses, actor2_losses = [],[],[]

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
        local_c_l,local_a1_l,local_a2_l = [],[],[]
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
            critic_loss, (actor1_loss,actor2_loss) = agent.step(state,action,reward,next_state,done, batch_size)
            local_c_l.append(critic_loss.detach().numpy())
            local_a1_l.append(actor1_loss.detach().numpy())
            local_a2_l.append(actor2_loss.detach().numpy())
            # update score
            score += np.array(env_info.rewards)
            #check if finished
            if np.any(done):
                break
            else:
                state = next_state

        # Update monitorization variables & params for next Episode
        critic_losses.append(np.max(np.array(local_c_l)))
        actor1_losses.append(np.max(np.array(local_a1_l)))
        actor2_losses.append(np.max(np.array(local_a2_l)))
        num_visits.append(visits)
        scores1.append(score[0])
        scores2.append(score[1])
        scores.append(np.max(score))
        scores_window.append(np.max(score))

        if e % print_stats_every == 0:
            output = 'Episode {} \tMax visits in last 10 episodes: {} \tLast 10 episodes: {:.5f}\tAvg Score: {:.5f}\tCritic Loss:{:.7f}\tA1 Loss:{:.5f}\tA2 Loss:{:.5f}'\
                  .format(e,np.max(num_visits[-10:]),np.mean(scores[-10:]),np.mean(scores_window),np.mean(critic_losses[-10:]),np.mean(actor1_losses[-10:]),np.mean(actor2_losses[-10:]))
            print(output)
            std_out += '\n' + output

        if e % print_scores_every == 0:
            plotResults(scores,0,compacted = False)

        if np.mean(scores_window) >= stop_criteria and aux:
            print('Environment solved in {} episodes'.format(e))
            aux = False
            checkpoint = e
            max_episodes = e + 200 # UPDATE THE NUMBER OF MAX_EPISODES

            weights_goal_checkpoint[0] = agent.actors[0].state_dict()
            weights_goal_checkpoint[1] = agent.actors[1].state_dict()
            weights_goal_checkpoint[2] = agent.critic.state_dict()

        if np.mean(scores_window) >= stop_criteria and np.mean(scores[-10:]) > best_score: # max score=2.5 --> if last10 ep avg more than 1.5 --> save weights
            best_score = np.mean(scores[-10:])
            checkpoint_best = e
            weights_best_checkpoint[0] = agent.actors[0].state_dict()
            weights_best_checkpoint[1] = agent.actors[1].state_dict()
            weights_best_checkpoint[2] = agent.critic.state_dict()

        # STOP CRITERIA
        if e >= max_episodes or np.mean(scores_window) > 1.8:
            break

    return scores,scores1,scores2,\
            checkpoint,checkpoint_best,\
            weights_goal_checkpoint,weights_best_checkpoint,\
            best_score,std_out,\
            critic_losses, actor1_losses, actor2_losses


def evaluate_agent(agent,env,brain_name,n_episodes=1):
    scores = []
    input()
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
    env, brain_name, state_size,action_size, num_agents = defineEnvironment(env_path,verbose=True)

    # define agent
    agent = MADDPG_Agent(state_size,action_size,num_agents,GAMMA, \
                         TAU, LR_ACTOR, LR_CRITIC, BUFFER_SIZE, \
                         BUFFER_TYPE, POLICY_UPDATE,\
                         NOISE_INIT, NOISE_DECAY, NOISE_MIN)

    if mode == 'train':

        scores,s1,s2,checkpoint,checkpoint_best,weights_checkpoint,best_weights,best_score,\
            std_out,critic_losses,actor1_losses,actor2_losses = \
            train_agent(agent,env,brain_name,n_episodes=EPISODES, batch_size = BATCH_SIZE)

        result_folder_path = generateTestFolder()

        # export used hyperparms
        with open(result_folder_path + '/hyperparameters.md', "w") as f:
            f.write('## HYPERPARAMETERS\n')
            f.write('\tEPISODES: {}\n'.format(EPISODES))
            f.write('\tGAMMA: {}\n'.format(GAMMA))
            f.write('\tTAU: {}\n'.format(TAU))
            f.write('\tLR_ACTOR: {}\n'.format(LR_ACTOR))
            f.write('\tLR_CRITIC: {}\n'.format(LR_CRITIC))
            f.write('\tBUFFER_SIZE: {}\n'.format(BUFFER_SIZE))
            f.write('\tBUFFER_TYPE: {}\n'.format(BUFFER_TYPE))
            f.write('\tBATCH_SIZE: {}\n'.format(BATCH_SIZE))
            f.write('\tPOLICY_UPDATE: {}\n'.format(POLICY_UPDATE))
            f.write('\n\tNORMAL/GAUSSIAN NOISE RELATED PARAMS\n')
            f.write('\tNOISE_INIT {}\n'.format(NOISE_INIT))
            f.write('\tNOISE_DECAY {}\n'.format(NOISE_DECAY))
            f.write('\tNOISE_MIN {}\n'.format(NOISE_MIN))
        # export training log
        with open(result_folder_path + '/training_log.md', "w") as f:
            f.write(std_out)
        # export/save models weights
        if checkpoint > 0:
            print('\nEnvironment solved in {} episodes'.format(checkpoint))
            torch.save(weights_checkpoint[0], result_folder_path + '/actor' + str(1) + '_model_weights_checkpoint.pth')
            torch.save(weights_checkpoint[1], result_folder_path + '/actor' + str(2) + '_model_weights_checkpoint.pth')
            torch.save(weights_checkpoint[2], result_folder_path + '/critic_model_weights_checkpoint.pth')
            if checkpoint_best > 0:
                print('\nBest_score: {:.2f} obtained in {} episode'.format(best_score,checkpoint_best))
                torch.save(best_weights[0], result_folder_path + '/actor' + str(1) + '_model_weights_best.pth')
                torch.save(best_weights[1], result_folder_path + '/actor' + str(2) + '_model_weights_best.pth')
                torch.save(best_weights[2], result_folder_path + '/critic_model_weights_best.pth')
        # export scores file
        with open(result_folder_path + '/scores','wb') as f:
            pickle.dump([scores,s1,s2,checkpoint,checkpoint_best],f)
        # export network losses file
        with open(result_folder_path + '/network_losses','wb') as f:
            pickle.dump([critic_losses,actor1_losses,actor2_losses],f)
        # export scores image
        compact_scores = scoresEvery100episodes(scores)
        fig = plotResults_overlap(scores,compact_scores,checkpoint)
        fig.savefig(result_folder_path + '/scores.png')

    elif mode == 'evaluation':
        w1 = 'results/test_1/actor1_model_weights_checkpoint.pth'
        # w1 = 'results/test_1/actor1_model_weights_best.pth'
        w2 = 'results/test_1/actor2_model_weights_checkpoint.pth'
        # w2 = 'results/test_1/actor2_model_weights_best.pth'
        agent.actors[0].load_state_dict(torch.load(w1))
        agent.actors[0].eval()
        agent.actors[1].load_state_dict(torch.load(w2))
        agent.actors[1].eval()
        scores = evaluate_agent(agent,env,brain_name,n_episodes=3)
        checkpoint = None
    # close env
    env.close()
