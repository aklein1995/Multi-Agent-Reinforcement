
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
    plt.close()
    # plt.show()

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

    # ax.set_xticks(np.append(ax.get_xticks(),checkpoint))
    # ax.set_xticklabels(np.append(ax.get_xticklabels(),checkpoint))
    # print(ax.get_xticks())
    # print(ax.get_xticklabels())
    plt.show()
    return fig

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
