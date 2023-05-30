import torch
import torch.nn as nn
import math
from functools import reduce
import numpy as np


import numpy as np
from pydec.dec import simplicial_complex
from pydec.math.circumcenter import weighted_circ, circumcenter, circumcenter_barycentric
import torch
# from pydec.math import circumcenter


class MyModel(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self,w):
        super().__init__()
        weights = torch.Tensor(w)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

    def forward(self, X,sc):
        num_bathches=X.shape[0]
        loss=[]
        sc.weights=self.weights
        M=sc[1].star
        for i in range(num_bathches):
            loss.append(torch.linalg.solve(M,X[i]))
            return loss
    
b=torch.tensor([0,1.,1],requires_grad=False)
v=np.array([[0,0],[1,0],[-1,1]])    
t=np.array([[0,1,2]])
w=torch.tensor([0,1.,1],requires_grad=True)
sc=simplicial_complex((v,t))
sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=torch.float32)

model=MyModel(w)

# loss function and optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
batches_per_epoch = len(Xtrain) // batch_size
 
for epoch in range(n_epochs):
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
 
# evaluate trained model with test set
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print("Accuracy {:.2f}".format(accuracy * 100))
