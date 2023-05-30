import numpy as np
import os
import pickle

import torch

from utils import *

path1='/Users/idanversano/Documents/clones/pydec/data/input/'
path2='/Users/idanversano/Documents/clones/pydec/data/output/'
pathes=[path1,path2]
for path in pathes:
  if not os.path.exists(path):
        os.makedirs(path)

v=np.array([[0,0],[1,0],[-1,1]])    
t=np.array([[0,1,2]])

f1,u1=create_data(v)
f2,u2=create_data(v)
u=[u1,u2]
f=[f1,f2]
for i in range(len(u)):
    pickle.dump(f[i], open(path1+'f'+str(i) , "wb"))
    pickle.dump(u[i], open(path2+'u'+str(i) , "wb"))



