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

    def forward(self, f,sc):
        sc.weights=self.weights
        M=sc[1].star
        return torch.linalg.solve(M,f)
    
b=torch.tensor([0,1.,1],requires_grad=False)
v=np.array([[0,0],[1,0],[-1,1]])    
t=np.array([[0,1,2]])
w=torch.tensor([0,1.,1],requires_grad=True)
sc=simplicial_complex((v,t))
sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=torch.float32)

model=MyModel(w)
model(b,sc)
