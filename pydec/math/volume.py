__all__ = ['unsigned_volume','signed_volume']

from scipy import sqrt,inner,shape,asarray
from scipy.special import factorial
from scipy.linalg import det
import math
import torch

def unsigned_volume(pts):
    # pts = asarray(pts) 

    
    M,N = pts.shape
    M -= 1

    if M < 0 or M > N:
        raise ValueError('array has invalid shape')
    
    if M == 0:
        return 1.0 
        
    A = pts[1:] - pts[0]
   
    return torch.sqrt(abs(torch.det(torch.inner(A,A))))/math.factorial(M)
    
    
def signed_volume(pts):
    
    
    # pts = asarray(pts)
    
    M,N = pts.shape
    M -= 1

    if M != N:
        raise ValueError('array has invalid shape')
        
    A = pts[1:] - pts[0]
    return det(A)/factorial(M)
