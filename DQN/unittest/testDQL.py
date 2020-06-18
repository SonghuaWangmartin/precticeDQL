# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:06:23 2020

@author: martin
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

import unittest
from ddt import ddt, data, unpack
from src.simpleDQL import trainmethod,Buildmodel,dqnmodel



@ddt
class TestsimpleDQL(unittest.TestCase):
    def setUp(self):
        statedim = 3
        actiondim = 2
        learningRate = 0.001
        numberlayers = 20
        self.states,self.Qvalue = Buildmodel(statedim,actiondim,numberlayers)
        self.action_,self.TargetQ_,self.optimizer = trainmethod(actiondim, self.Qvalue, 
                                                                learningRate)
        
    @data(([0.4, 1.3, -0.2, 1.5], 0),
        ([-0.1, 0.4, 0.1, 0.6], 1)
        )
         
    @unpack
    def testMaxaction(self,states,resultaction):
        expectaction = maxaction(states)
        self.assertEqual(expectaction, resultaction)
if __name__ == '__main__':
    unittest.main()
