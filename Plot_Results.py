import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

GOAL = 0.5

def subplot_results(scores,compacted_scores,checkpoint=0):
    fig, (ax, ax2) = plt.subplots(1,2)

    ax.plot(np.arange(len(scores)), scores)
    ax.axhline(y=GOAL, linewidth = 1, color = 'black', linestyle = '--')
    ax.set_title('Scores obtained in each episode')
    ax.set(xlabel='# Episode',ylabel='Score')
    
    ax2.plot(np.arange(len(compacted_scores)), compacted_scores)
    ax2.axhline(y=GOAL, linewidth = 1, color = 'black', linestyle = '--')
    if checkpoint > 0:
        best_score = np.max(compacted_scores) + 1
        ax2.set_ylim([0,best_score])
        ax2.axvline(x=checkpoint,ymax = GOAL/best_score,linewidth = 1, color = 'red', linestyle = ':')
    ax2.set_title('Avg score obtained in 100 consecutive episodes')
    ax2.set(xlabel='# Episode',ylabel='Score')
    
    plt.show()

def plotResults(scores,checkpoint=0,compacted = True):
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

def plotResults_overlap(scores,compacted_scores,checkpoint=0):
    fig, ax = plt.subplots(1,1)

    best_score = np.max([scores,compacted_scores]) + 0.5
    ax.set_ylim([0,best_score])
    
    ax.plot(np.arange(len(scores)), scores,label='raw')
    ax.plot(np.arange(len(compacted_scores)), compacted_scores,label='avg 100 episodes',color='#FF7F0E')
    ax.axhline(y=GOAL, linewidth = 1, color = 'black', linestyle = '--')
    ax.legend()
    ax.set_title('Scores obtained in each episode')
    ax.set(xlabel='# Episode',ylabel='Score')
    
    if checkpoint > 0:
        ax.axvline(x=checkpoint, ymax = GOAL/best_score,linewidth = 2, color = 'red', linestyle = ':')
    
    plt.show()
    
def scoresEvery100episodes(scores):
    compact_scores = []
    for i in range(1,len(scores)+1):
        if i < 100:
            compact_scores.append(np.mean(scores[0:i]))
        else:
            compact_scores.append(np.mean(scores[i-100:i]))
    return compact_scores

def analyzeSingleRun(filename):
    with open(filename,'rb') as f:
       scores,s1,s2,checkpoint,checkpoint_best = pickle.load(f)
    
    compact_scores = scoresEvery100episodes(scores)
    
    subplot_results(scores,compact_scores,checkpoint)
    plotResults_overlap(scores,compact_scores,checkpoint)
    plotResults(scores,checkpoint,compacted = False)
    plotResults(compact_scores,checkpoint)
    
    print('Problem solved in {} episodes'.format(checkpoint))
    print('Best performance avg {} obtained in {} episodes'.format(np.max(compact_scores),checkpoint_best))    
    print('After being solved, each episode has:')
    print('-Avg score: {}\n-std: {}'.format(np.mean(scores[checkpoint:]),np.std(scores[checkpoint:])))
    
if __name__ == "__main__": 
    
    filename = 'results/test_5/scores' 
    analyzeSingleRun(filename)
    with open(filename,'rb') as f:
       scores,s1,s2,checkpoint,checkpoint_best = pickle.load(f)
    with open('results/test_5/network_losses','rb') as f:
        critic_losses,actor1_losses,actor2_losses = pickle.load(f)

    plt.plot(actor1_losses,label='actor1')
    plt.plot(actor2_losses,label='actor2')
    plt.plot(critic_losses,label='critic')
    plt.legend()
    plt.xlabel('# episode')
    plt.title('Losses')
    plt.show()
    
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel('# Episodes')
    ax.set_ylabel('actor losses')
    plt1 = ax.plot(actor1_losses,label='actor1',color='blue')
    plt2 = ax.plot(actor2_losses,label='actor2',color='orange')
    
    ax2 = ax.twinx()
    ax2.set_ylabel('critic loss')
    plt3 = ax2.plot(critic_losses,label='critic',color='green')
    
    #labels
    plots = plt1 + plt2 + plt3
    labs = [l.get_label() for l in plots]
    ax.legend(plots,labs)
    
    #colors yaxis tunned
    # ax.yaxis.label.set_color('#FF0000')
    # ax.tick_params(axis='y',colors='#FF0000')
    ax2.yaxis.label.set_color('green')
    ax2.tick_params(axis='y',colors='green')
    # *************************************************************************
    p1_winning = (np.array(s1) >= np.array(s2)).astype(int)
    p2_winning = (np.array(s2) >= np.array(s1)).astype(int)
    p_draw = (np.array(s2) == np.array(s1)).astype(int)
    
    init,fin = 800,850   
    fig, (ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,10))
    ax.plot(p1_winning[init:fin],label='p1',alpha=0.5)
    ax.set_xticks(np.arange(np.abs(fin-init)))
    ax.set_xticklabels(np.arange(init,fin),rotation=45)
    ax.set_ylabel('Player 1 wins')

    ax2.plot(p2_winning[init:fin],label='p2',alpha=0.5)
    ax2.set_xticks(np.arange(np.abs(fin-init)))
    ax2.set_xticklabels(np.arange(init,fin),rotation=45)
    ax2.set_ylabel('Player 2 wins')
    
    ax3.plot(p_draw[init:fin],alpha=0.5)
    ax3.set_xticks(np.arange(np.abs(fin-init)))
    ax3.set_xticklabels(np.arange(init,fin),rotation=45)
    ax3.set_ylabel('Both players draw')
    plt.show()
    