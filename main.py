"""
Darcy flow in a triangle mesh of a planar domain, with constant
velocity prescribed on boundary. 

Reference :

Numerical method for Darcy flow derived using Discrete Exterior Calculus
A. N. Hirani, K. B. Nakshatrala, J. H. Chaudhry
See arXiv:0810.3434v3 [math.NA] on http://arxiv.org/abs/0810.3434

"""
import math
import numpy as np
import sys
# sys.path.append('/Users/idanversano/Documents/clones2/pydec/pydec')
from pydec.pydec.math import *

# pydec.math.circumcenter.circumcenter(pts)
# vertices=np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5],[0.5,0.6]])
# triangles=np.array([[0,1,4],[1,2,4], [4,3,0],[4,2,5],[4,5,3],[2,3,5]])
# f=[]
# u=[]
# sc = sim((vertices,triangles)) 

# circumcenter(vertices)

# def weighted_circ(pts,weights):
#     s=pts[0]
#     if len(pts)==2:
#             n=(pts[1]-pts[0])/unsigned_volume(pts)
#             s=s+(1/(2*unsigned_volume(pts)))* (
#             unsigned_volume([pts[0],pts[1]])**2+weights[0]-weights[1]
#             )*n
#     if len(pts)==3:
#         A=np.array([[0,-1],[1,0]])
#         n1=np.dot(A,pts[0]-pts[2])
#         n2=np.dot(A,pts[1]-pts[0])
#         n=[n1,n2]
#         k=2
#         for i in range(k):
#             s=s+(1/(2*math.factorial(k)*unsigned_volume(pts)))* (
#             unsigned_volume([pts[0],pts[i+1]])**2+weights[0]-weights[i+1]
#             )*n[i]   
#     return s
# pts=asarray([[0,0],[4,1],[0,4]])
# weights=[0.2,0.3,0.5]
# triangles=asarray([[0,1,2]])
# figure(); ax = gca()
# ax.triplot(pts[:,0], pts[:,1], triangles)
# # print(np.dot(weighted_circ([pts[1],pts[2]],[weights[1],weights[2]])-weighted_circ(pts, weights),pts[1]-pts[2]))
# print(circumcenter(pts))
    

# def objective(x,*args):
#     A=args[0]
#     b=args[1]
#     y =A@x-b
 
   
#     return np.linalg.norm(y)



# # Read the mesh



# for v in vertices:
#     f.append(-2*math.pi**2*np.cos(math.pi*v[0])*np.cos(math.pi*v[1]))
#     u.append(np.cos(math.pi*v[0])*np.cos(math.pi*v[1]))
# f=np.array(f)   
# u=np.array(u)
# cons = ({'type': 'eq', 'fun': lambda x: x.sum() - u.sum()})

# # Make a simplicial complex from it

# figure(); ax = gca()
# ax.triplot(vertices[:,0], vertices[:,1], triangles)
# # show()
# # # Nk is number of k-simplices
# N1 = sc[1].num_simplices
# N2 = sc[2].num_simplices
# print(N1)
# d0 = sc[0].d; 
# star1 = sc[1].star; 


# star0= sc[0].star
# star2=sc[2].star

# star0_inv=star0.copy()
# print(sc.embedding_dimension())
# print(N1)
# print(len(sc[0].bary_circumcenter))
# print(sc[2].bary_circumcenter)
# for i in range(star0.shape[0]):
#     star0_inv[i,i]=1/star0[i,i]
# M=star0_inv@(-d0.transpose())@star1@d0

# # res = optimize.minimize(objective, u*0+2, args=(M,f), method='SLSQP', constraints=cons, options={'disp': False})
# # print((res['x'])[-2:]-u[-2:])


# # print(1/star0)
# # print(sc[0].primal_volume)
# # print(sc[0].dual_volume)
# # print(whitney_innerproduct (sc,1).todense())
# # print(sc[1].primal_volume)
# # print(star1.todense())
# # # Permeability is k > 0 and viscosity is mu > 0
# k = 1; mu = 1
# # # The matrix for the full linear system for Darcy in 2D, not taking into
# # # account the boundary conditions, is :
# # # [-(mu/k)star1 d1^T ]
# # # [    d1          Z ] 
# # # where Z is a zero matrix of size N2 by N2. 
# # # The block sizes are 
# # #   N1 X N1    N1 X N2
# # #   N2 X N1    N2 X N2


# A = bmat([[(-mu/k)*sc[1].star, sc[1].d.T],
#           [sc[1].d, None]], format='csr')

# # b = zeros(N1 + N2) # RHS vector
# # all_fluxes = zeros(N1)
# # # Find boundary and internal edges
# # boundary_edges = sc.boundary(); boundary_edges.sort()
# # boundary_indices = list(sort([sc[1].simplex_to_index[e] 
# #                                        for e in boundary_edges]))
# # num_boundary_edges = len(boundary_indices)
# # internal_edges = set(sc[1].simplex_to_index.keys()) - set(boundary_edges)
# # internal_indices = list(sort([sc[1].simplex_to_index[e] 
# #                                        for e in internal_edges]))
# # num_internal_edges = sc[1].num_simplices - num_boundary_edges
# # # Assume triangles oriented the same way so can look at any triangle
# # s = sign(det(vertices[triangles[0,1:]] - vertices[triangles[0,0]]))
# # for i, e in enumerate(boundary_edges):
# #     evector = (-1)**e.parity * (vertices[e[1]] - vertices[e[0]])
# #     normal = array([-evector[1], evector[0]])
# #     all_fluxes[boundary_indices[i]] = -s * (1/norm(evector)**2) * \
# #                               dot(velocity, normal) * \
# #                               abs(det(vstack((normal, evector))))
# # pressures = zeros(N2)
# # # Adjust RHS for known fluxes and pressures
# # b = b - A * concatenate((all_fluxes,pressures))
# # # Remove entries of b corresponding to boundary fluxes and known pressure
# # # Pressure at the right most triangle circumcenter is assumed known
# # pressure_indices = list(range(N1, N1+N2))
# # pressure_indices.remove(N1 + argmax(sc[2].circumcenter[:,0]))
# # entries_to_keep = concatenate((internal_indices, pressure_indices))
# # b = b[entries_to_keep]
# # # Remove corresponding rows and columns of A
# # A = A[entries_to_keep][:,entries_to_keep]
# # u = spsolve(A,b)
# # fluxes = u[0:len(internal_indices)]
# # pressures[array(pressure_indices)-N1] = u[len(internal_indices):]
# # # Plot the pressures
# # figure(); ax = gca();
# # ax.set_xlabel('x', fontsize=20)
# # ax.set_ylabel('Pressure', fontsize=20)
# # ax.plot(sc[2].circumcenter[:,0], pressures, 'ro', markersize=8, mec='r')
# # # Draw a line of slope -1 through the known presure point
# # xmax = max(sc[2].circumcenter[:,0])
# # xmin = min(sc[2].circumcenter[:,0])
# # ax.plot([xmin, xmax], [xmax - xmin, 0], 'k', linewidth=2)
# # ax.legend(['DEC', 'Analytical'], numpoints=1)
# # # Plot the triangles

# # # Insert the computed fluxes into vector of all fluxes
# # all_fluxes[internal_indices] = fluxes
# # # Whitney interpolate flux and sample at barycenters. Then
# # # rotate 90 degrees in the sense opposite to triangle orientation
# # v_bases,v_arrows = simplex_quivers(sc, all_fluxes)
# # v_arrows = -s * vstack((-v_arrows[:,1], v_arrows[:,0])).T
# # # Plot the resulting vector field at the barycenters
# # ax.quiver(v_bases[:,0],v_bases[:,1],v_arrows[:,0],v_arrows[:,1],
# #        units='dots', width=1, scale=1./30)
# # ax.axis('equal')
# # ax.set_title('Flux interpolated using Whitney map \n' \
# #           ' and visualized as velocity at barycenters\n')

# # show()
