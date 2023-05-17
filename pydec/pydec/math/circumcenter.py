__all__ = ['is_wellcentered', 'circumcenter', 'circumcenter_barycentric']

from numpy import bmat, hstack, vstack, dot, sqrt, ones, zeros, sum, \
        asarray
from numpy.linalg import solve,norm
import numpy as np
import math
from scipy.linalg import det
from scipy import sqrt,inner

def unsigned_volume(pts):
    
    pts = asarray(pts)
    
    M,N = pts.shape
    M -= 1

    if M < 0 or M > N:
        raise ValueError('array has invalid shape')
    
    if M == 0:
        return 1.0 
        
    A = pts[1:] - pts[0]
    return sqrt(abs(det(inner(A,A))))/math.factorial(M)




def is_wellcentered(pts, tol=1e-8):
    """Determines whether a set of points defines a well-centered simplex.
    """
    barycentric_coordinates = circumcenter_barycentric(pts)    
    return min(barycentric_coordinates) > tol

def circumcenter_barycentric(pts, weights=None):
    # pts = asarray(pts)

    # rows,cols = pts.shape

    # assert(rows <= cols + 1)    

    # A = bmat( [[ 2*dot(pts,pts.T), ones((rows,1)) ],
    #            [  ones((1,rows)) ,  zeros((1,1))  ]] )

    # b = hstack((sum(pts * pts, axis=1),ones((1))))
    # x = solve(A,b)
    # bary_coords = x[:-1]  

    # return bary_coords
    



    pts = asarray(pts)
    rows,cols = pts.shape
    assert(rows <= cols + 1) 

    if rows==1:
        bary_coords=np.array([1.])
    if rows==2:
        p1=pts[0,:]
        p2=pts[1,:]

        center=np.array(circumcenter(pts,weights)[0])
        rhs=center-p2
        lhs=p1-p2
        if abs(lhs[0])<1e-10:
            l1=rhs[1]/lhs[1]
        else:
            l1=rhs[0]/lhs[0]
    
        bary_coords=np.array([l1,1-l1])

    if rows==3:
        center=np.array(circumcenter(pts, weights)[0])
        A = bmat( [[ pts.T], [ones((1,rows))] ]
              )

        b = hstack((np.array(center),np.array([1])))

        x = solve(A,b)

        bary_coords = x 
    return bary_coords
    
    
def circumcenter(pts,weights):
    
    pts = asarray(pts)      
    # bary_coords = circumcenter_barycentric(pts)
    # center = dot(bary_coords,pts)
    center=weighted_circ(pts,weights)
    radius = norm(pts[0,:] - center)
    return (center,radius)
    
    
    
def weighted_circ(pts,weights):

    s=pts[0]
    if len(pts)==2:
            n=(pts[1]-pts[0])/unsigned_volume(pts)
            s=s+(1/(2*unsigned_volume(pts)))* (
            unsigned_volume([pts[0],pts[1]])**2+weights[0]-weights[1]
            )*n
    if len(pts)==3:
        
        

        A=np.array([[0,-1],[1,0]])
        n1=np.dot(A,pts[0]-pts[2])
        n2=np.dot(A,pts[1]-pts[0])
        n=[n1,n2]
        k=2
        for i in range(k):
            s=s+(1/(2*math.factorial(k)*unsigned_volume(pts)))* (
            unsigned_volume([pts[0],pts[i+1]])**2+weights[0]-weights[i+1]
            )*n[i]   
        
    return s
    