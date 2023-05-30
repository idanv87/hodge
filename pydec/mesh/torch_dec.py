import numpy as np
from pydec.dec import simplicial_complex

v=np.array([[0,0],[1,0],[1,1]])
t=np.array([[0,1,2]])

sc=simplicial_complex((v,t))
print(sc[0].d)
