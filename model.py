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
# sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=torch.float32)
# sc[0].d=torch.tensor(sc[0].d.todense(), requires_grad=False, dtype=torch.float32)
w0=np.random.rand(len(v))*0
# w0[boundary_ind]=0
w=torch.tensor(w0,requires_grad=True, dtype=Constants.dtype)

# plot_mesh(v,t,None,boundary_ind)
# sc=simplicial_complex((v,t))

# sc.weights=w

# sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=torch.float32)
# sc[0].d=torch.tensor(sc[0].d.todense(), requires_grad=False, dtype=torch.float32)



f,u=load_data()
Xtrain, Xtest, Ytrain, Ytest = train_test_split( f, u, test_size=0.1, random_state=42)
Constants.batch_size=len(Xtrain)
Xtrain=batch_divide(Xtrain, Constants.batch_size)
Ytrain=batch_divide(Ytrain, Constants.batch_size)
# # u=torch.reshape(torch.tensor(np.array(u),dtype=torch.float32), np.array(u).shape)
# # f=torch.reshape(torch.tensor(np.array(f),dtype=torch.float32), np.array(f).shape)
# Xtrain=batch_divide(f, Constants.batch_size)
# Ytrain=batch_divide(u, Constants.batch_size)
# # Xtrain=f
# # Ytrain=u
# # Xtrain=[f]
# # Ytrain=[u]
  

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
        
        # print(self.weights)
        sc.weights=self.weights
      
        sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=Constants.dtype)
        sc[0].d=torch.tensor(sc[0].d.todense(), requires_grad=False, dtype=Constants.dtype)

        M=-sc[0].star_inv@(-(sc[0].d).T)@sc[1].star@sc[0].d
        # print(M.requires_grad)
       
        # M=(-(sc[0].d).T)@sc[0].d
        # print(M.shape)
        # print(torch.linalg.matrix_rank(M))
        
        M=M[self.interior_indices][:,self.interior_indices]+torch.eye(len(self.interior_indices))
        # print(torch.linalg.matrix_rank(M))
        # print(M.shape)
        # print(torch.linalg.matrix_rank(M))
        # plot_mesh(v,t)
        res=[]
      
        for i in range(batch_size):
                     res.append(torch.linalg.solve(M,X[i,self.interior_indices]))
                     
                 
        return torch.reshape(torch.cat(res, dim=0),(batch_size,M.shape[0]))
        # return 


# loss_fn = nn.MSELoss()
model=MyModel(w)

# Xbatch = Xtrain[0]
# ybatch = Ytrain[0]
# print(ybatch)
# print(ybatch.shape)
# forward pass
# y_pred = model(Xbatch)
# print(abs(model(Xbatch)@ybatch[0,model.interior_indices].T-Xbatch[0,model.interior_indices].T))

# loss =loss_fn(y_pred, ybatch[:,model.interior_indices])
# print(ybatch[:,model.interior_indices]-y_pred)
loss_fn = nn.MSELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_batches=len(Xtrain)
for epoch in range(20):
    
    for i in range(num_batches):
      
        # take a batch
        Xbatch = Xtrain[i]
        Ybatch = Ytrain[i]
        # forward pass
        optimizer.zero_grad()
        y_pred = model(Xbatch)
        loss =loss_fn(y_pred, Ybatch[:,model.interior_indices])
        # print(loss.requires_grad)
     
        loss.backward()
        optimizer.step()
   
        print(loss)
        
        
# f,u=create_data(v,1,1)
# # u=torch.reshape(torch.tensor(u,dtype=torch.float32), (1,len(u)))
# # f=torch.reshape(torch.tensor(f,dtype=torch.float32), (1,len(f)))
# sc=simplicial_complex((v,t))
# sc.weights=w0*0
# sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=Constants.dtype)
# sc[0].d=torch.tensor(sc[0].d.todense(), requires_grad=False, dtype=Constants.dtype)

# M=sc[0].star_inv@(-(sc[0].d).T)@sc[1].star@sc[0].d
# print(torch.argmin(sc[0].star))
# print(sc[1].star[138,138])

# print(sc[0].d[138])
# print(np.linalg.norm(v[70]-v[45]))

# # print(sc[0].star[0,0])
# choosen=interior_ind
# M=M[:,choosen][choosen]

# # print(torch.linalg.matrix_rank(torch.tensor(M.todense())))
# # print(M.shape)
# # print((M*dx**2).todense())
# err=abs(M@u[choosen]-f[choosen])
# print(err)
# print(M[0,0])

# print(choosen)



        # model.sc.weights=model.weights
        # print(model.sc.weights)
    
 
# # evaluate trained model with test set
# with torch.no_grad():
#     y_pred = model(X)
# accuracy = (y_pred.round() == y).float().mean()
# print("Accuracy {:.2f}".format(accuracy * 100))
