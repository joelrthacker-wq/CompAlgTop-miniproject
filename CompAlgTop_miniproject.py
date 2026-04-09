import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import itertools
import numpy as np

#Chatgpt's code for generating a point cloud and computing vietoris rips

def random_point_cloud(n_points, dim):
    rng = np.random.default_rng()  
    return rng.random((n_points, dim))


def pairwise_distances(points):
    n = len(points)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(points[i] - points[j])
            d[i, j] = d[j, i] = dij
    return d


def simplex_birth(simplex, dist_matrix):
    """
    Vietoris-Rips birth time of a simplex is the maximum pairwise distance
    among its vertices.
    """
    if len(simplex) <= 1:
        return 0.0
    return max(dist_matrix[i, j] for i, j in itertools.combinations(simplex, 2))


def vietoris_rips_complex(points, max_dim):
    n = len(points)
    dist_matrix = pairwise_distances(points)

    # All simplices up to max_dim
    simplices = []
    for k in range(1, max_dim + 2):  # size 1 means dim 0, size 2 means dim 1, etc.
        for simplex in itertools.combinations(range(n), k):
            birth = simplex_birth(simplex, dist_matrix)
            simplices.append((simplex, birth))

    # Obtain filtration steps:
    thresholds = sorted(set(birth for _, birth in simplices))

    F_vector_spaces = {}
    for simplex, birth in simplices:
        l_sigma = [1 if birth <= t else 0 for t in thresholds]
        F_vector_spaces[simplex] = l_sigma

    return thresholds, F_vector_spaces

#My very rough code for converting this into the cosheaf data required for filtered_coscythe
    
def random_simplicial_complex(n_points, max_dim):
    points = random_point_cloud(n_points=n_points, dim=5)
    thresholds, F_vector_spaces = vietoris_rips_complex(points, max_dim=max_dim) 
    K = list(F_vector_spaces.keys())
    filt_length = len(F_vector_spaces[K[0]])
    F_maps = [{}]
    K_minus = {x : [] for x in K}
    K_plus = {x : [] for x in K}
    connecting_maps = []
    for x in K:
        if len(x) > 1:
            x_minus = []
            for i in range(len(x)):
                y = x[:i] + x[i+1:]
                x_minus.append(y)
                K_plus[y].append(x)
                d_x = F_vector_spaces[x][-1]
                d_y = F_vector_spaces[x][-1]
                r_xy = np.eye(d_y, d_x)
                #r_xy = np.random.randn(d_x, d_y)
                F_maps[0][(x,y)] = r_xy
            K_minus[x] = x_minus
        
    for i in range(0,filt_length-1):
        connecting_map_new = {}
        for x in K:
            d_x = F_vector_spaces[x][i]
            d_x_plus = F_vector_spaces[x][i+1]
            connecting_map_new[x] = np.eye(d_x_plus, d_x)
        connecting_maps.append(connecting_map_new)

    for i in range(1,filt_length)[::-1]:
        K_map_new = {}
        for key, value in F_maps[0].items():
            x = key[0]
            y = key[1]
            d_x = F_vector_spaces[x][i-1]
            d_y = F_vector_spaces[y][i-1]
            r_xy = value
            r_xy_new =  r_xy[:d_y, :d_x]
            K_map_new[(x,y)] = r_xy_new
        F_maps.insert(0, K_map_new)
    
    F_vector_spaces_new = [{} for i in range(filt_length)]
    for key, value in F_vector_spaces.items():
        for i in range(len(value)):
            F_vector_spaces_new[i][key] = value[i]
        

    return K, K_minus, K_plus, F_vector_spaces_new, F_maps, connecting_maps


def filtered_coscythe(K, F_vector_spaces, F_maps, connecting_map, K_minus=None, K_plus=None):
    """
    Input:
    K is a list of tuples. It represents a graded poset, where elements of dimension k are tuples of length k, and the order is determined by K_minus and K_plus. If these 
    are not input, the ordering is assumed to be the face relation.
    F_vector_spaces is a list of dictionaries, with F_vector_spaces[i][sigma] = dim(F_{i+1}(sigma))
    F_maps is a list of dictionaries, with F[i][(tau, sigma)] being the matrix of F_{i+1, tau sigma}
    connecting_maps is a list of dictionaries, with connecting_maps[i][sigma] being the matrix of delta_{i+1, sigma}

    Output:
    K backslash Sigma
    F_vector_spaces^{Sigma}
    F_maps^{Sigma}
    connecting_maps
    >_{Sigma}, via the dictionaries K_minus and K_plus

    Where Sigma is an acyclic partial matching on K
    """
    def ReducePair(x,y, x_minus_modified):
        for beta in x_minus_modified:
            y_plus_modified = K_plus[y].copy()
            y_plus_modified.remove(x)
            for alpha in y_plus_modified:
                # Set alpha \triangleright beta
                K_plus[beta].append(alpha)
                K_minus[alpha].append(beta)
                # Update F_{n, \alpha \beta}
                if (alpha,beta) in F_maps[-1]:
                    r_alphabeta = F_maps[-1][(alpha,beta)]
                else:
                    r_alphabeta = np.zeros((F_vector_spaces[-1][beta], F_vector_spaces[-1][alpha]))
                F_maps[-1][(alpha,beta)] = r_alphabeta - F_maps[-1][(x, beta)] @ np.linalg.inv(F_maps[-1][(x,y)]) @ F_maps[-1][(alpha, y)]
            
        # Remove x,y from K
        for a in K_minus[x]:
            K_plus[a].remove(x)
        for b in K_plus[x]:
            K_minus[b].remove(x)
        for a in K_minus[y]:
            K_plus[a].remove(y)
        for b in K_plus[y]:
            K_minus[b].remove(y)
        K.remove(x)
        K.remove(y)

        return K
    
    # If K_minus, K_plus are not input, we assume the ordering is given by the face relation and compute K_minus, K_plus
    if K_minus == None and K_plus == None:
        K_minus = {x : [] for x in K}
        K_plus = {x : [] for x in K}
        for x in K:
            if len(x) > 1:
                x_minus = []
                for i in range(len(x)):
                    y = x[:i] + x[i+1:]
                    x_minus.append(y)
                    K_plus[y].append(x)
                K_minus[x] = x_minus

    filtration_length = len(F_vector_spaces)

    # Almost line for line the algorithm presented in the miniproject
    queue = deque()
    K_noncritical = K.copy()
    while K_noncritical != []:
        c = max(K_noncritical, key = len)
        K_noncritical.remove(c)
        queue.clear()
        queue.append(c)
        S = []
        while (queue) and S != K:
            sigma = queue.popleft()
            S.append(sigma)
            sigmaplus_noncritical = list(set(K_plus[sigma]) & set(K_noncritical))
            if len(sigmaplus_noncritical) == 1:
                tau = sigmaplus_noncritical[0]
                # Checking if F_{i, \tau \sigma} is invertible for all i
                compatible = True
                for i in range(filtration_length):
                    if F_maps[i][(tau, sigma)].shape[0] != F_maps[i][(tau, sigma)].shape[1]:
                        compatible = False
                        break
                    else:
                        if np.linalg.det(F_maps[i][(tau, sigma)]) == 0:
                            compatible = False
                            break
                if compatible:
                    tau_minus_modified = K_minus[tau].copy()
                    tau_minus_modified.remove(sigma)
                    for x in tau_minus_modified:
                        if len(x) == len(c)-1:
                            queue.append(x)
                    # We add tau_minus_modified as an input to ReducePair because we already have it to hand
                    K = ReducePair(tau, sigma, tau_minus_modified)
            sigma_minus = K_minus[sigma]
            for x in sigma_minus:
                if len(x) == len(c)-1:
                    queue.append(x)
    for tau in K:
        for sigma in K_minus[tau]:
            for i in range(1, filtration_length)[::-1]:
                delta_i_sigma_inv = np.linalg.inv(connecting_maps[i-1][sigma].T @ connecting_maps[i-1][sigma]) @ connecting_maps[i-1][sigma].T
                F_maps[i-1][(tau, sigma)] = delta_i_sigma_inv @ F_maps[i][(tau, sigma)] @ connecting_maps[i-1][tau]

    return K, F_vector_spaces, F_maps, connecting_maps, K_minus, K_plus



max_dim = 10
n_points = 10
K, K_minus, K_plus, F_vector_spaces, F_maps, connecting_maps = random_simplicial_complex(n_points = n_points, max_dim = max_dim)
print('Initial simplicial complex: \n', K, '\n\n\n')
K, F_vector_spaces, F_maps, connecting_maps, K_minus, K_plus = filtered_coscythe(K, F_vector_spaces, F_maps, connecting_maps)
print('Critical simplices: \n', K, '\n')


