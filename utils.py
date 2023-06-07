import math

import numpy as np
from matplotlib.pyplot import gca, show
from scipy.spatial import Delaunay
import pickle
from numpy import loadtxt
import torch

from constants import Constants

path1='/Users/idanversano/Documents/clones/pydec/data/input/'
path2='/Users/idanversano/Documents/clones/pydec/data/output/'

def bc(v):
    bc_indices=[]

    for i,ver in enumerate(v):
       if (1 in set(ver)) or  (0 in set(ver)):
          bc_indices.append(i)   
    return bc_indices   

def create_data(vertices,kx,ky):
    k=kx**2+ky**2
  
    f=[]
    u=[]
    for v in vertices:
      f.append((1+k*math.pi**2)*np.sin(kx*math.pi*v[0])*np.sin(ky*math.pi*v[1]))
      u.append(np.sin(kx*math.pi*v[0])*np.sin(ky*math.pi*v[1]))
      # u.append(v[0]**2+v[1])
      # f.append(2.)
    return np.array(f), np.array(u)  

def plot_mesh(v,t,sc=None, ind=None):
    colors=['red','black']
    ax = gca()
    ax.triplot(v[:,0], v[:,1], t)

    if ind is not None:
            
             [ax.scatter(v[i][0],v[i][1],c=colors[0]) for i in ind]

    
    if sc is not None :
     for dim in [1,2]:
    
      [ax.scatter(v[0].detach().numpy(),v[1].detach().numpy(),c=colors[dim-1]) for v in sc[dim].circumcenter]
    show()

def triangulate(points):
    tri = Delaunay(points)    
    # plt.triplot(points[:,0], points[:,1], tri.simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.show()
    return tri.simplices      

def create_vertices(n):
    x,y=np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
    v=[]
    bc_indices=[]
    for i in range(len(x)):
       for j in range(len(y)):
          v.append([x[i,j],y[i,j]])
    for i,ver in enumerate(v):
       if (1 in set(ver)) or  (0 in set(ver)):
          bc_indices.append(i)   
    return np.array(v), bc_indices      
          
def create_mesh(path='/Users/idanversano/Documents/clones/pydec/examples/DarcyFlow/'):
  v = loadtxt(path+'vertices.txt')
  t = loadtxt(path+'triangles.txt', dtype='int') -1
  a=min([V[0] for V in v])
  b=max([V[0] for V in v])
  v = (loadtxt('vertices.txt'))*(1/(b-a))-a/(b-a)
  boundary_indices=bc(v)
  interior_indices=list(set(range(len(v)))-set(boundary_indices))
  return v,t,boundary_indices, interior_indices

def load_data():
   with open(path1+'f.pickle', 'rb') as f:
    X = pickle.load(f)
   with open(path2+'u.pickle', 'rb') as u:
    Y = pickle.load(u)
   return X,Y  
# for i in boundary_indices:
#     ax = gca()
#     ax.scatter(v[i][0],v[i][1])

def batch_divide(x, batch_size):
  
   n=int(len(x)/batch_size)
  
   X=[]
   for j in range(n):
      u=x[j*batch_size:(j+1)*batch_size]
      X.append(torch.reshape(torch.tensor(np.array(u),dtype=Constants.dtype), np.array(u).shape))

   return X


   


# f,u=create_data(v)
# # u=torch.reshape(torch.tensor(u,dtype=torch.float32), (1,len(u)))
# # f=torch.reshape(torch.tensor(f,dtype=torch.float32), (1,len(f)))
# sc=simplicial_complex((v,t))
# M=sc[0].star_inv@(-(sc[0].d).T)@sc[1].star@sc[0].d
# M=M[:,choosen][choosen]
# print(torch.linalg.matrix_rank(torch.tensor(M.todense())))
# print(M.shape)
# print((M*dx**2).todense())
# err=abs(M@u-f)
# print(err[choosen])

# print(u)
# D=((-sc[0].d).T@sc[0].d).todense()
# print(D[choosen[10]])
# print(D[0])
# print(v[0])
# print(sc[1].star)
# print(sc[0].d.shape)
# plot_mesh(v,t)
# choosen=list(set(range(len(v)))-set(boundary_indices))
# M=M[choosen][:,choosen]
# print(M@u-f)
# print(np.max(sc[0].star_inv))
# print((sc[0].d).todense())



# print(v)
# vertices = loadtxt('vertices.txt')
# triangles = loadtxt('triangles.txt', dtype='int') - 1
# plot_mesh(vertices, triangles)

# v=np.array([[-1,0],[0,0],[1,0],[0,1]])    
# t=np.array([[0,1,3],[1,2,3]])
# plot_mesh(v,t)