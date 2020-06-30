# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:14:08 2020

@author: martin
"""


import os
import sys


dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
from src.simpleDQL import UpdateParameters,plotrewards,epsilonDec,BuildModel,ReplayMemory,TrainModel,TrainDQNmodel,Getaction
from env.Cartpole import CartPoleEnvSetup,visualizeCartpole,resetCartpole,CartPoletransition,CartPoleReward,isTerminal
from collections import deque



gamma = 0.9
buffersize = 20000
batchsize =  32 
epsilon = 0.9
minepsilon = 0.2
epsilondec = 0.01
learningRate = 0.01
numberlayers = 15
replaceiter = 200


EPISODE = 10000
STEP = 300 
test = 10

def main():
    env = CartPoleEnvSetup()        
    visualize = visualizeCartpole() 
    reset = resetCartpole()         
    transition = CartPoletransition()  
    rewardcart = CartPoleReward()        
    isterminal = isTerminal()
    
    statesdim = env.observation_space.shape[0]
    actiondim = env.action_space.n
    replaybuffer = deque()
    runepsilon = epsilon
    
    buildmodel = BuildModel(statesdim,actiondim)
    Writer,DQNmodel = buildmodel(numberlayers)
    getaction = Getaction(actiondim)
    updateParameters = UpdateParameters(replaceiter)
    trainModel = TrainModel(learningRate, gamma,Writer)
    trainDQNmodel = TrainDQNmodel(updateParameters, trainModel, DQNmodel)
    replayMemory = ReplayMemory(buffersize,batchsize,trainDQNmodel)

    
    for episode in range(EPISODE):
        state  = reset()
        for step in range(STEP):
            runepsilon = epsilonDec(runepsilon,minepsilon,epsilondec)
            action = getaction(DQNmodel,runepsilon,state)  
            nextstate=transition(state, action)
            done = isterminal(nextstate)
            rewards = rewardcart(done)
            replayMemory(replaybuffer,state, action, rewards, nextstate)
            state = nextstate
            if done:
                break
        if episode % 100 == 0:
            totalrewards = 0   # total reward
            for i in range(test):
                state = reset()
                for j in range(STEP):
                    visualize(state)
                    action = getaction(DQNmodel,epsilon,state)       
                    nextstate=transition(state, action)
                    done = isterminal(nextstate)
                    reward = rewardcart(done)
                    state=nextstate
                    totalrewards += reward
                    if done:
                        break
            meanreward = totalrewards/test
            print('episode: ',episode,'average Reward:',meanreward)
            if meanreward >= 200:
                visualize.close()
                break
            plotrewards(rewards, meanreward)
if __name__ == '__main__':
  main()
