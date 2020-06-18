#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:23:24 2020

@author: martin
"""
import os
import sys

dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName,   '..'))
import unittest
from ddt import ddt, data, unpack
from src.simpleDQL import Buildmodel

@ddt
class TestNNstructure(unittest.TestCase):
    @data(([4],[1],[10] ,[[[None,4]]],[[[None,1]]])
          )
    @unpack
    def testBuildodel(self, statedim,actiondim,numlayers,groundstatestrueShapes,groundtargettrueShapes):
        states_,TargetQ = Buildmodel(statedim,actiondim,numlayers)
        generatedstatesShapes = [states_.shape.as_list()]
        generatedTargetQShapes = [TargetQ.shape.as_list()]
        self.assertEqual(generatedstatesShapes, groundstatestrueShapes)
        self.assertEqual(generatedTargetQShapes, groundtargettrueShapes)

if __name__ == '__main__':
    unittest.main()