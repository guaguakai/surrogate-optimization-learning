# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 08:16:46 2019

@author: Aditya
"""

import networkx as nx 
import numpy as np

source=0
target=6
G1=nx.Graph([(source,1),(source,2),(1,2),(1,3),(1,4),(2,4),(2,5),(4,5),(3,target),(4,target),(5,target)], source=0, target=6)
#nx.draw(G1)


G2=nx.Graph([(0,1),(1,2),(1,3),(1,3),(2,4),(2,5),(4,8),(5,8),(3,6),(3,7),(6,9),(7,9),(8,10),(9,10),(10,11)])
nx.draw(G2)