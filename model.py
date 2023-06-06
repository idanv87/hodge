import torch
import torch.nn as nn
import math
from functools import reduce
import numpy as np


import numpy as np
from pydec.dec import simplicial_complex
from pydec.math.circumcenter import weighted_circ, circumcenter, circumcenter_barycentric
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from sklearn.model_selection import train_test_split

from constants import Constants

path1='/Users/idanversano/Documents/clones/pydec/data/input/'
path2='/Users/idanversano/Documents/clones/pydec/data/output/'
# from pydec.math import circumcenter

from utils import *

v,t,boundary_ind,interior_ind=create_mesh()
w0=np.random.rand(len(v))*0
w=torch.tensor(w0,requires_grad=True, dtype=torch.float32)

# sc=simplicial_complex((v,t))

# sc.weights=w

# sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=torch.float32)
# sc[0].d=torch.tensor(sc[0].d.todense(), requires_grad=False, dtype=torch.float32)



f,u=load_data()
Xtrain, Xtest, Ytrain, Ytest = train_test_split( f, u, test_size=0.33, random_state=42)

# u=torch.reshape(torch.tensor(np.array(u),dtype=torch.float32), np.array(u).shape)
# f=torch.reshape(torch.tensor(np.array(f),dtype=torch.float32), np.array(f).shape)
Xtrain=batch_divide(Xtrain, Constants.batch_size)
Ytrain=batch_divide(Ytrain, Constants.batch_size)
# print(len(Xtrain))
# Xtrain=[f]
# Ytrain=[u]

    

class MyModel(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self,weight, interior_indices=interior_ind):
        super().__init__()
        self.weights = nn.Parameter(weight)  
        # self.interior_indices=interior_indices
        self.v,self.t,self.boundary_ind,self.interior_indices=create_mesh()

    
       

    def forward(self, X):
        batch_size=X.shape[0]
        sc=simplicial_complex((self.v,self.t))
        
       
        sc.weights=self.weights
        sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=torch.float32)
        sc[0].d=torch.tensor(sc[0].d.todense(), requires_grad=False, dtype=torch.float32)
        M=sc[0].star_inv@(-(sc[0].d).T)@sc[1].star@sc[0].d
        M=M[self.interior_indices][:,self.interior_indices]
        res=[]
        for i in range(batch_size):
                     res.append(torch.linalg.solve(M,X[i,self.interior_indices]))
                 
        return torch.reshape(torch.cat(res, dim=0),(batch_size,M.shape[0]))



model=MyModel(w)


loss_fn = nn.MSELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_batches=len(Xtrain)
for epoch in range(2):
    
    for i in range(num_batches):
      
        # take a batch
        Xbatch = Xtrain[i]
        ybatch = Ytrain[i]
        # forward pass
        optimizer.zero_grad()
        y_pred = model(Xbatch)
        loss =loss_fn(y_pred, ybatch[:,model.interior_indices])
        loss.backward()
        optimizer.step()
        



        # model.sc.weights=model.weights
        # print(model.sc.weights)
    
 
# # evaluate trained model with test set
# with torch.no_grad():
#     y_pred = model(X)
# accuracy = (y_pred.round() == y).float().mean()
# print("Accuracy {:.2f}".format(accuracy * 100))
