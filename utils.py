import math
import numpy as np

def create_data(vertices):
    f=[]
    u=[]
    for v in vertices:
      f.append(-2*math.pi**2**2*np.cos(math.pi*v[0])*np.cos(math.pi*v[1]))
      u.append(np.cos(math.pi*v[0])*np.cos(math.pi*v[1]))
    #   u.append(v[0]+2*v[1])
    return np.array(f), np.array(u)  
