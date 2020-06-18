# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 06:25:29 2020

@author: martin
"""

import tensorflow.compat.v1 as tf
import numpy as np 
import random
from collections import deque
tf.enable_v2_behavior()
tf.compat.v1.disable_eager_execution()


def trainmethod(actiondim,Qvalue,learningRate):
    action_ = tf.placeholder("float",[None,actiondim])
    TargetQ_ = tf.placeholder("float",[None])
    TrainQ = tf.reduce_sum(tf.multiply(Qvalue,action_),reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(TargetQ_ - TrainQ))
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
    return action_,TargetQ_,optimizer
    
def Buildmodel (statedim,actiondim,numberlayers):
    graph = tf.Graph()
    with graph.as_default():
        initweight = tf.random_normal_initializer(0, 0.3)
        initbias = tf.constant_initializer(0.1)
    with tf.name_scope('inputs'):
        states_ = tf.placeholder("float", [None, statedim])  
        tf.add_to_collection('states_', states_)
    with tf.variable_scope('net1'):
        weight1 = tf.get_variable("w1", [statedim, numberlayers], initializer=initweight)
        bias1 = tf.get_variable("b1", [1, numberlayers], initializer=initbias)
        layer1 = tf.nn.relu(tf.matmul(states_, weight1) + bias1)
        tf.add_to_collection('weight1', weight1)
        tf.add_to_collection('bias1', bias1)
        tf.add_to_collection('layer1', layer1)
    with tf.variable_scope('layer2'):
        weight2 = tf.get_variable("w2", [numberlayers, actiondim], initializer=initweight)
        bias2 = tf.get_variable("b2", [1, actiondim], initializer=initbias)
        Qtarget = tf.matmul(layer1, weight2) + bias2
        tf.add_to_collection('weight2', weight2)
        tf.add_to_collection('bias2', bias2)
        tf.add_to_collection('Qtarget', Qtarget)
    return states_,Qtarget
    
class dqnmodel():
    def __init__(self,statesdim,actiondim,fixedparameter):
        self.gamma = fixedparameter['gamma']
        self.learningRate = fixedparameter['learningRate']
        self.epsilon = fixedparameter['epsilon']
        self.numberlayers = fixedparameter['numberlayers']
        self.replaysize = fixedparameter['replaysize']
        self.batchsize = fixedparameter['batchsize']
        self.replaybuffer = deque()
        
        self.statedim = statesdim
        self.actiondim = actiondim
        self.states,self.Qvalue = Buildmodel(self.statedim,self.actiondim,self.numberlayers)
        self.action_,self.TargetQ_,self.optimizer = trainmethod(self.actiondim, self.Qvalue, 
                                                                self.learningRate)
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        
    
    def Updatemodel (self,state, actions, rewards, nextStates,done):
        encodeaction = np.zeros(self.actiondim) 
        encodeaction[actions] = 1
        self.replaybuffer.append((state, encodeaction, rewards, nextStates, done))
        if len(self.replaybuffer) > self.replaysize:
            self.replaybuffer.popleft()
        if len(self.replaybuffer) > self.batchsize:
            minibatch = random.sample(self.replaybuffer,self.batchsize)
            stateBatch = [D[0] for D in minibatch]
            actionbatch = [D[1] for D in minibatch]
            rewardBatch = [D[2] for D in minibatch]
            nextStateBatch = [D[3] for D in minibatch]
            Qvaluebatch = self.Qvalue.eval(feed_dict={self.states: nextStateBatch})
            ybatch = []
            for i in range(0, self.batchsize):
                done = minibatch[i][4]
                if done:
                    ybatch.append(rewardBatch[i])
                else:
                    ybatch.append(rewardBatch[i] + self.gamma * np.max(Qvaluebatch[i]))
            
            self.optimizer.run(feed_dict={self.TargetQ_: ybatch,self.action_: actionbatch,self.states: stateBatch})

    def getaction(self,state):
        Qvalue  = self.Qvalue.eval(feed_dict = {self.states:[state]})[0]
        if random.random() <= self.epsilon:
            action = random.randint(0,self.actiondim-1)  
        else:
            action =  np.argmax(Qvalue)
        return action
    
    def maxaction(self,state):
        Qvalue = self.Qvalue.eval(feed_dict = {self.states:[state]})[0]
        return np.argmax(Qvalue)
                    
               
