import numpy as np
from pydec.dec import simplicial_complex
from pydec.math.circumcenter import weighted_circ, circumcenter, circumcenter_barycentric
import torch

b=torch.tensor([0,1.,1],requires_grad=False)
v=np.array([[0,0],[1,0],[-1,1]])

t=np.array([[0,1,2]])
w=torch.tensor([0,1.,1],requires_grad=True)
# print(torch.hstack((torch.tensor(v)[0],torch.tensor(v)[1])))
# c=weighted_circ(torch.tensor(v),w)
# c=circumcenter_barycentric(torch.tensor(u),w)
# print(c)
sc=simplicial_complex((v,t))
sc.vertices=torch.tensor(sc.vertices, requires_grad=False, dtype=torch.float32)
sc.weights=w
# print(sc.simplices)
x=torch.norm(torch.linalg.solve(sc[1].star,b))

# print(sc[1].primal_volume)
# print(sc[1].dual_volume)

