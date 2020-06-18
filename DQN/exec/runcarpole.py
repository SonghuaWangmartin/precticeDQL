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
from src.simpleDQL import dqnmodel
from env.Cartpole import CartPoleEnvSetup,visualizeCartpole,resetCartpole,CartPoletransition,CartPoleReward,isTerminal



fixedparameters={
  'gamma':0.8,
  'replaysize':20000,
  'batchsize': 32 ,
  'epsilon' : 0.4,
  'learningRate':0.01,
  'numberlayers':15
  }


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
  DQNmodel = dqnmodel(statesdim,actiondim,fixedparameters)
  for episode in range(EPISODE):
    # initialize task
    state = reset()
    # Train
    for step in range(STEP):
      action = DQNmodel.getaction(state)  
      next_state=transition(state, action)
      done = isterminal(next_state)
      reward = rewardcart(done)
      DQNmodel.Updatemodel(state, action, reward, next_state,done)
      state = next_state
      if done:
        break
    if episode % 100 == 0:
      totalrewards = 0   # total reward
      for i in range(test):
        state = reset()
        for j in range(STEP):
          visualize(state)
          action = DQNmodel.maxaction(state)      # direct action for test
          next_state=transition(state, action)
          done = isterminal(next_state)
          reward = rewardcart(done)
          state=next_state
          totalrewards += reward
          if done:
            break
      meanreward = totalrewards/test
      print('episode: ',episode,'average Reward:',meanreward)
      if meanreward >= 200:
        visualize.close()
        break
if __name__ == '__main__':
  main()
