import numpy as np
import os
import pickle

import torch

from utils import *
from sklearn.model_selection import train_test_split

path1='/Users/idanversano/Documents/clones/pydec/data/input/'
path2='/Users/idanversano/Documents/clones/pydec/data/output/'
pathes=[path1,path2]
for path in pathes:
  if not os.path.exists(path):
        os.makedirs(path)


v,t,boundary_ind,interior_ind=create_mesh()
kx=[1]
ky=kx
u=[]
f=[]
for k1 in kx:
    F,U=create_data(v,k1,k1)
    u.append(U)
    f.append(F)

pickle.dump(f, open(path1+'f.pickle' , "wb"))
pickle.dump(u, open(path2+'u.pickle' , "wb"))
# for i in range(len(u)):
#     pickle.dump(f[i], open(path1+'f'+str(i) , "wb"))
#     pickle.dump(u[i], open(path2+'u'+str(i) , "wb"))


