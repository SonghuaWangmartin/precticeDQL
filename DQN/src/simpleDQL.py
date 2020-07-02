# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 06:25:29 2020

@author: martin
"""
import tensorflow.compat.v1 as tf
import numpy as np 
import matplotlib.pyplot as plt
import random
tf.enable_v2_behavior()
tf.compat.v1.disable_eager_execution()

class ReplaceParameters:
    def __init__(self, replaceiter):
        self.replaceiter = replaceiter
    def __call__(self, model,runtime):
        if runtime % self.replaceiter == 0:
            modelgraph = model.graph
            replaceParam_ = modelgraph.get_collection_ref("ReplaceTargetParam_")[0]
            model.run(replaceParam_)
        return model

def samplebuffer(replaybuffer,batchsize):
    minibatch = random.sample(replaybuffer,batchsize)
    return minibatch

def epsilonDec(epsilon,minepsilon,epsilondec):
    runepsilon = epsilon - epsilondec if epsilon > minepsilon else minepsilon
    return runepsilon

class BuildModel ():
    def __init__(self, statedim, actiondim):
        self.stateDim = statedim
        self.actionDim = actiondim
        
    def __call__(self, numberlayers):
        graph = tf.Graph()
        with graph.as_default():
            initweight = tf.random_normal_initializer(0, 0.3)
            initbias = tf.constant_initializer(0.1)
            with tf.name_scope('inputs'):
                states_ = tf.placeholder(tf.float32, [None, self.stateDim])
                nextstates_ = tf.placeholder(tf.float32, [None, self.stateDim])
                reward_ = tf.placeholder(tf.float32, [None,])
                action_ = tf.placeholder(tf.float32, [None, self.actionDim])
                tf.add_to_collection('states_', states_)
                tf.add_to_collection('nextstates_', nextstates_)
                tf.add_to_collection("reward_", reward_)
                tf.add_to_collection("action_", action_)
            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                gamma_ = tf.constant(0, dtype=tf.float32)
                tf.add_to_collection("learningRate_", learningRate_)
                tf.add_to_collection("gamma_", gamma_)
            with tf.variable_scope('evalnet'):
                with tf.variable_scope('layer1'):
                    Weight1 = tf.get_variable("w1", [self.stateDim, numberlayers], initializer=initweight)
                    Bias1 = tf.get_variable("b1", [1, numberlayers], initializer=initbias)
                    layer1 = tf.nn.relu(tf.matmul(states_, Weight1) + Bias1)
                    tf.add_to_collection('Weight1', Weight1)
                    tf.add_to_collection('Bias1', Bias1)
                    tf.add_to_collection('layer1', layer1)
                with tf.variable_scope('layer2'):
                    Weight2 = tf.get_variable("w2", [numberlayers, self.actionDim], initializer=initweight)
                    Bias2 = tf.get_variable("b2", [1, self.actionDim], initializer=initbias)
                    Qevalvalue_ = tf.matmul(layer1, Weight2) + Bias2
                    tf.add_to_collection('Weight2', Weight2)
                    tf.add_to_collection('Bias2', Bias2)
                    tf.add_to_collection('Qevalvalue_', Qevalvalue_)
                    
            with tf.variable_scope('targetnet'):
                with tf.variable_scope('layer1'):
                    weight1 = tf.get_variable("w1", [self.stateDim, numberlayers], initializer=initweight)
                    bias1 = tf.get_variable("b1", [1, numberlayers], initializer=initbias)
                    layer1 = tf.nn.relu(tf.matmul(nextstates_, weight1) + bias1)
                    tf.add_to_collection('weight1', weight1)
                    tf.add_to_collection('bias1', bias1)
                    tf.add_to_collection('layer1', layer1)
                with tf.variable_scope('layer2'):
                    weight2 = tf.get_variable("w2", [numberlayers, self.actionDim], initializer=initweight)
                    bias2 = tf.get_variable("b2", [1, self.actionDim], initializer=initbias)
                    Qnext = tf.matmul(layer1, weight2) + bias2
                    tf.add_to_collection('weight2', weight2)
                    tf.add_to_collection('bias2', bias2)
                    tf.add_to_collection('Qnext', Qnext)   
                
            with tf.variable_scope('Qtarget'):
                qtarget = reward_ + gamma_ * tf.reduce_max(Qnext, axis=1)   
                Qtarget =  tf.stop_gradient(qtarget)
                tf.add_to_collection("Qtarget", Qtarget)
            
            with tf.variable_scope('Qevalaction'):
                Qevalaction = tf.reduce_sum(tf.multiply(Qevalvalue_,action_),reduction_indices = 1) 
                tf.add_to_collection("Qevalaction", Qevalaction)
                
            with tf.variable_scope('loss'):
                loss_ = tf.reduce_mean(tf.squared_difference(Qtarget, Qevalaction))
                tf.add_to_collection("loss_", loss_)
                
            with tf.variable_scope('train'):
                trainopt = tf.train.RMSPropOptimizer(learningRate_, name='adamOptimizer').minimize(loss_)
                tf.add_to_collection("trainopt", trainopt)
                
            with tf.name_scope("replaceParameters"):
                evalParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evalnet')
                targetParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetnet')
                ReplaceTargetParam_ = [tf.assign(targetParams_, evalParams_) for targetParams_, evalParams_ in zip(targetParams_, evalParams_)]
                tf.add_to_collection("evalParams_", evalParams_)
                tf.add_to_collection("targetParams_", targetParams_)
                tf.add_to_collection("ReplaceTargetParam_", ReplaceTargetParam_) 
                
            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)

            model = tf.Session(graph = graph)
            model.run(tf.global_variables_initializer())
            
            Writer = tf.summary.FileWriter('/path/to/logs', graph = graph)
            tf.add_to_collection("Writer", Writer)
        return Writer, model
    
    
class TrainModel:
    def __init__(self, learningRate, gamma,Writer):
        self.learningRate = learningRate
        self.gamma = gamma
        self.Writer = Writer
    def __call__(self, model, minibatch):
        statebatch = [d[0] for d in minibatch]
        actionbatch = [d[1] for d in minibatch]
        rewardbatch = [d[2] for d in minibatch]
        nextStatebatch = [d[3] for d in minibatch]
        modelGraph = model.graph
        states_ = modelGraph.get_collection_ref("states_")[0]
        action_ = modelGraph.get_collection_ref("action_")[0]
        reward_ = modelGraph.get_collection_ref("reward_")[0]
        nextstates_ = modelGraph.get_collection_ref("nextstates_")[0]
        learningRate_ = modelGraph.get_collection_ref("learningRate_")[0]
        gamma_ = modelGraph.get_collection_ref("gamma_")[0]

        loss_ =  modelGraph.get_collection_ref("loss_")[0]
        trainopt = modelGraph.get_collection_ref("trainopt")[0]
        
        model.run([trainopt,loss_],feed_dict={states_: statebatch, action_: actionbatch, reward_: rewardbatch, nextstates_: nextStatebatch,
                                      learningRate_: self.learningRate, gamma_: self.gamma})

        summary = tf.Summary()
        summary.value.add(tag='reward', simple_value=float(np.mean(rewardbatch)))
        self.Writer.flush()
        
        return model
    
class TrainDQNmodel():
    def __init__(self, updateParameters, trainModel, DQNModel):
        self.updateParameters = updateParameters
        self.trainModel = trainModel
        self.DQNModel = DQNModel
        self.runtime = 0
    def __call__(self, miniBatch):
        self.DQNModel = self.updateParameters(self.DQNModel,self.runtime)
        self.trainModel(self.DQNModel, miniBatch)
        self.runtime += 1
        
class Learn():
    def __init__(self,replaysize,batchsize,trainDQNmodel,actiondim):
        self.replaysize = replaysize
        self.batchsize = batchsize
        self.trainDQNmodel = trainDQNmodel
        self.actionDim = actiondim
    def ReplayMemory(self,replaybuffer,state, action, rewards, nextstate):
        onehotaction = np.zeros(self.actionDim)
        onehotaction[action] = 1
        replaybuffer.append((state, onehotaction, rewards, nextstate))
        if len(replaybuffer) > self.replaysize:
            replaybuffer.popleft()
        if len(replaybuffer) > self.batchsize:
            minibatch = samplebuffer(replaybuffer,self.batchsize)
            self.trainDQNmodel(minibatch)                          
            
    def Getaction(self,model,epsilon,state):
        modelGraph = model.graph
        Qevalvalue_ = modelGraph.get_collection_ref("Qevalvalue_")[0]
        states_ = modelGraph.get_collection_ref("states_")[0]
        Qvalue = model.run(Qevalvalue_, feed_dict={states_: [state]})
        if random.random() > epsilon:
            action = np.argmax(Qvalue)
        else:
            action = random.choice(np.arange(self.actionDim))
        return action
      




def plotrewards(rewards, avgrewards):
    """
    Plot rewards and running average rewards.
    Args:
        rewards: list of rewards 
        avgrewards: list of average (last 100) rewards
    """
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(2, 1, 2)
    plt.subplots_adjust(hspace=.5)
    
    ax1.set_title('Running rewards')
    ax1.plot(avgrewards, label='average rewards')
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Average rewards")
    
    plt.show(fig)