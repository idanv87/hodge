__all__ = ['is_wellcentered', 'circumcenter', 'circumcenter_barycentric']

from numpy import bmat, hstack, vstack, dot, sqrt, ones, zeros, sum, \
        asarray
from numpy.linalg import solve,norm
import numpy as np
import math
from scipy.linalg import det
from scipy import sqrt,inner
import torch


from constants import Constants

def orient(a,b,c):
      return torch.sign((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
     
def unsigned_volume(pts):
    # pts = asarray(pts) 

    
    M,N = pts.shape
    M -= 1

    if M < 0 or M > N:
        raise ValueError('array has invalid shape')
    
    if M == 0:
        return 1.0 
        
    A = pts[1:] - pts[0]

    if sqrt(abs(det(torch.inner(A,A))))/math.factorial(M)<1e-5:
         print('error')
    return sqrt(abs(det(torch.inner(A,A))))/math.factorial(M)




def is_wellcentered(pts, tol=1e-8):
    """Determines whether a set of points defines a well-centered simplex.
    """
    barycentric_coordinates = circumcenter_barycentric(pts)    
    return min(barycentric_coordinates) > tol

def circumcenter_barycentric(pts, weights):

    rows,cols = pts.shape
    assert(rows <= cols + 1) 

    if rows==1:
        bary_coords=torch.tensor([1], dtype=torch.float64)
    if rows==2:
        p1=pts[0,:]
        p2=pts[1,:]

        center=circumcenter(pts,weights)[0]
        rhs=center-p2
        lhs=p1-p2
        if abs(lhs[0])<1e-10:
            l1=rhs[1]/lhs[1]
        else:
            l1=rhs[0]/lhs[0]
    
        bary_coords=torch.hstack((l1,1-l1))

    if rows==3:
        center=circumcenter(pts, weights)[0]
        
        A=torch.vstack((pts.T,torch.ones((1,rows), dtype=torch.float64)))
        # A = bmat( [[ pts.T], [ones((1,rows))] ]
            #   )
    
        b = torch.hstack((center,torch.tensor([1], dtype=torch.float64)))
        

        x = torch.linalg.solve(A,b)

        bary_coords = x 
        
    return bary_coords

    



    
    
def circumcenter(pts,weights=None):
    
    # pts = asarray(pts)
        
    # bary_coords = circumcenter_barycentric(pts)
    # center = dot(bary_coords,pts)
    center=weighted_circ(pts,weights)



    radius = torch.norm(pts[0,:] - center)
    return (center,radius)
    
    
    
def weighted_circ(pts,weights):

    s=pts[0]
    if len(pts)==2:
            n=(pts[1]-pts[0])/unsigned_volume(pts)
            s=s+(1/(2*unsigned_volume(pts)))* (
            unsigned_volume(
                   torch.vstack((pts[0],pts[1]))
                  )**2+weights[0]-weights[1]
            )*n
    if len(pts)==3:
        if orient(pts[0],pts[1],pts[2])==0:
             print('stop')
        
        
        if orient(pts[0],pts[1],pts[2])>0:
                  A=torch.tensor([[0,-1],[1,0]], dtype=Constants.dtype)
        else:  
                  A=torch.tensor([[0,-1],[1,0]], dtype=Constants.dtype).T 
        
        
        n1=torch.matmul(A,pts[0]-pts[2])
        n2=torch.matmul(A,pts[1]-pts[0])
        n=[n1,n2]
      
        k=2
        for i in range(k):
           
            s=s+(1/(2*math.factorial(k)*unsigned_volume(pts)))* (
            unsigned_volume(
                torch.vstack((pts[0],pts[i+1]))
                )**2+weights[0]-weights[i+1]
            )*n[i]   

 
    return s
    