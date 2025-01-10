# uufullerene.py module

# Anke ist wirklich toll
# test test
import numpy as np
import networkx as nx
import scipy.special as sc
import matplotlib.pyplot as plt
import math 
import os
import networkx as nx

path_database = "/Volumes/StochaExt/fullerene_database/buckygen"
path_buckygen = "/Users/arturbille/Documents/forschung/BuckyGen/buckygen"
path_planarread = "/Users/arturbille/Documents/forschung/BuckyGen/planarread_mod"


################################################# Constants ################################################# 

### Golden Ratio ###
golden_ratio = (1 + 5 ** 0.5) / 2

################################################# Basic functions ################################################# 

### Binomial coefficient with special cases ### 
def binom(n, k):
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    if k == 0 or n <= 1:
        return 1
    return binom(n - 1, k) + binom(n - 1, k - 1)

### Multinomial coefficient using function binom ####
def multinomial(params):
    #Summary: Compute the multinomial coefficient k!/(k_1!*k_2!*k_3!*....*k_len(params)!) using recursion formula for
    #         factorial
    #Input: params=[k_1,...,k_len(params)]
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])

### Factorial with special case of negative argument ###
def factorial(k):
    if k<0:
        return np.inf
    else: 
        return math.factorial(k)

### Check whether a matrix A is symmetric ###
def is_symmetric(A):
    if np.sum(A==np.transpose(A))==A.shape[0]**2:
        return 1
    else:
        return 0

########################################## Hexagonal and triangular lattices ###########################################################################
### Create the adjacency matrix A of a triangulation, without loops, where vertices are enumerated according to the spiral method ####
def create_A(size_param, method="spiral"):
    # r = number of rings around origin
    # 1+3*r*(r+1) is the number of vertices in this cutout
    method_list = ["LLUR","spiral"]
    if not method in method_list:
        print("Wrong method. Please, use one of the following methods:")
        for i in range(len(method_list)):
            print(method_list[i])
        return -1
    
    ### Lower-left-to-upper-right (LLUR) method
    if method=="LLUR":
        A = np.zeros((size_param**2,size_param**2))
        for row_index in range(size_param):
            #Declare the vertices on the very left-hand side and right-hand side 
            start_index = row_index * size_param
            end_index = (row_index + 1) * size_param - 1
            #print(start_index,end_index)
            #Connect in current row all vertices horizontally
            for i in range(start_index,end_index):
                A[i,i+1] = 1
                A[i+1,i] = 1
            if row_index < size_param - 1:
                #Connect current row with the row above
                start_index_next = (row_index + 1) * size_param
                end_index_next = (row_index + 2) * size_param -1
                #Connect first and last vertices of current row with above row
                #1.Case: Even rows
                if np.mod(row_index,2) == 0:
                    #print(start_index,start_index_next)
                    #print(end_index,end_index_next)
                    A[start_index,start_index_next] = 1
                    A[start_index_next,start_index] = 1
                    A[end_index,end_index_next - 1] = 1
                    A[end_index_next - 1,end_index] = 1
                    A[end_index,end_index_next] = 1
                    A[end_index_next,end_index] = 1
                    #Connect inner vertices
                    for i in range(1,size_param - 1):
                        #print(start_index + i,start_index_next + i - 1)
                        A[start_index + i,start_index_next + i - 1] = 1
                        A[start_index_next + i - 1,start_index + i] = 1
                        #print(start_index + i, start_index_next + i)
                        A[start_index + i, start_index_next + i] = 1
                        A[start_index_next + i,start_index + i] = 1            
                #2.Case: Odd rows
                if np.mod(row_index,2) == 1:
                    A[start_index,start_index_next] = 1
                    A[start_index_next,start_index] = 1
                    A[start_index,start_index_next + 1] = 1
                    A[start_index_next + 1,start_index] = 1
                    A[end_index,end_index_next] = 1
                    A[end_index_next,end_index] = 1
                    for i in range(1,size_param - 1):
                        #print(start_index + i,start_index_next + i)
                        A[start_index + i,start_index_next + i] = 1
                        A[start_index_next + i,start_index + i] = 1
                        #print(start_index + i, start_index_next + i + 1)
                        A[start_index + i, start_index_next + i + 1] = 1  
                        A[start_index_next + i + 1,start_index + i] = 1
        return A
    ### spiral method                
    if method=="spiral":
        if size_param < 1:
            print("Error: Argument size_param must be a positive integer.")
            return -1
        elif 1 <= size_param:
            num_ver = 1+3*size_param*(size_param+1)
            A = np.zeros((num_ver,num_ver))
            # Connect origin and first ring
            A[0,np.arange(1,7)] = 1
            A[np.arange(1,7),0] = 1
            indices = np.arange(1,7,1)
            A[indices[0:-1],indices[1:]] = 1
            A[indices[1:],indices[0:-1]] = 1
            A[indices[0],indices[-1]] = 1
            A[indices[-1],indices[0]] = 1
            
            #For every ring, starting form the second one (around the origin), do...
            for i in range(1,size_param):
                #Get indeces of ring i and i+1
                indices = np.arange(1+3*(i-1)*i,1+3*(i-1)*i+i*6, step = 1)
                indices_next = np.arange(1+3*i*(i+1),1+3*i*(i+1)+(i+1)*6, step = 1)
                #Connect all vertices within ring i...
                A[indices[0:-1],indices[1:]] = 1
                A[indices[1:],indices[0:-1]] = 1
                A[indices[0],indices[-1]] = 1
                A[indices[-1],indices[0]] = 1
                # ... and ring i+1
                A[indices_next[0:-1],indices_next[1:]] = 1
                A[indices_next[1:],indices_next[0:-1]] = 1
                A[indices_next[0],indices_next[-1]] = 1
                A[indices_next[-1],indices_next[0]] = 1
                
                # Connect the current ring with the next ring
                A[indices[-1],indices_next[0]] = 1
                A[indices_next[0],indices[-1]] = 1
                j = 0
                for ii in range(len(indices)):
                    deg_ii = np.sum(A,axis=0)[indices[ii]]
                    while deg_ii < 6:
                        A[indices[ii],indices_next[j]] = 1
                        A[indices_next[j],indices[ii]] = 1
                        deg_ii += 1
                        j += 1
                    j -= 1
        return A

### Shift a given adjacency matrix A by p, where p is the shiftvalue wrt. the x-axis on the triangular lattice
def A_shifted(A,p):
    #Total number of vertices
    n = np.shape(A)[0]
    #Number of rows/columns in the lattice cutout
    nrow = int(np.sqrt(n))
    #print(nrow)
    #Order of permuted columns 
    p_vec = np.zeros(n)
    i = 0
    for row in range(nrow):
        end_index = (row + 1) * nrow - 1
        for j in range(row * nrow + p,end_index + 1):
            p_vec[i] = j
            i += 1
        for j in range(row * nrow, row * nrow + p):
            p_vec[i] = j
            i += 1
    #Permutation matrix
    P = np.zeros((n,n))
    for i in range(n):
        P[i,int(p_vec[i])] = 1
    return np.matmul(A,P)
    
################################ Adjacency and Degree matrices of diffferent graphs and lattices ################################## 

    

### Compute linear combination alpha*A+beta*D of a adjacency matrix A ### 
def L(A,alpha = 1,beta = 0.5):
    return alpha*A + beta*np.diag(np.sum(A,axis=0))    

################################################# Densities of random eigenvalues ################################################# 
### Density function of T* (i.e., the triangulation with loops of weight 3)
def fT(x):
    if 0<x and x<9:
        return np.sqrt(3)/(np.pi*(3+x))*sc.hyp2f1(1/3, 2/3, 1, (x*(9-x)**2)/(3+x)**3)
    else:
        return 0

################################################# Moments #################################################
################################################# Direct way to compute the sum of squared multinomials #################################################
def a(k):
    result = 0
    for k1 in range(k+1):
        for k2 in range(k-k1+1):
            result += multinomial([k1,k2,k-k1-k2])**2
    return result

### Compute the kth moment of H (random eigenvalue of the hexagonal lattice) ###
def moment_H(k):
    if np.mod(k,2)==1:
        return 0
    else:
        result=0
        for k1 in range(int(k/2+1)):
            for k2 in range(int(k/2-k1+1)):
                result += multinomial([k1,k2,k/2-k1-k2])**2
        return result

### Compute the kth moment of H_tilde (normalized on [0,1] random eigenvalue of the hexagonal lattice) ###
def moment_H_tilde(k):
    result=0
    for j in range(k+1):
        result += 1/6**k*binom(k,j)*3**(k-j)*moment_H(j)
    return result

### Compute the kth moment of T (random eigenvalue of the triangular lattice) ###
def moment_T(k):
    result=0
    for k1 in range(k+1):
        for k2 in range(k-k1+1):
            result += multinomial([k1,k2,k-k1-k2])**2
    return result

### Compute the kth moment of T (random eigenvalue of the triangular lattice) normalized and centralized on [0,1] ###
def moment_T_tilde(k):
    return 1/9**k*moment_T(k)    

  
def f(k1,k):
    return 2*math.factorial(k-k1)**2/math.factorial(2*(k-k1))*binom(2*k1,k1+5)*hyper([1,(5-k1)/5,(6-k1)/5,(7-k1)/5,(8-k1)/5,(9-k1)/5],[(6+k1)/5,(7+k1)/5,(8+k1)/5,(9+k1)/5,(10+k1)/5],-1)



################################################# Dual nanotubes #################################################
################################################# Construction of adjacency matrix #################################################

def ring(r):
    if r<3:
        print("A ring needs at least three vertices, i.e., r>2.")
        return -1
    result = np.zeros((r,r))
    for i in range(r-1):
        result[i,i+1] = 1
        result[i+1,i] = 1
    result[0,-1] = 1
    result[-1,0] = 1
    return result

def ring_connection(r):
    result = np.eye(r)
    for i in range(1,r):
        result[i,i-1] = 1
    result[0,-1] = 1
    return result

def tube(r,n):
    if n<2:
        return ring(r)
    else:
        R = ring(r)
        RC = ring_connection(r)
        result = np.block([
            [R, RC],
            [np.transpose(RC),R]
        ])
        for i in range(3,n+1):
            result = np.block([[result,np.block([[np.zeros((np.shape(result)[0]-r,r))],
                                               [RC]])],
                              [np.block([[np.zeros((r,np.shape(result)[1]-r)),np.transpose(RC),R]])]]
                             )
    return result

################################################# Moments #################################################
### Compute the k-th moment of a dual (p,q)--nanotube ###
def moment_dual_nanotube(p,q,k):
    result = 0
    for k1 in range(k + 1):
        for k2 in range(k - k1 + 1):
            k3 = k - k1 - k2
            inner_sum = 0
            for z in range(1,int(np.floor(k1/(p+q))+1)):
                inner_sum += binom(2*k1-z*q,k1+z*p)*binom(k,k1-z*q)/binom(k,k1)/binom(2*(k-k1),k-k1)
            result += multinomial([k1,k2,k3])**2*(1+2*inner_sum)
    return int(np.round(result))
    
### Compute the k-th moment of a dual (p,q)--nanotube normalized and centralized on [0,1] ###
def moment_dual_nanotube_tilde(p,q,k):
    return 1/9**k*moment_N_pq(p,q,k)
    
def moment_N_50_1(k):
    result = 0
    for k1 in range(k+1):
        inner_sum=0
        for z in range(1,int(np.floor(k1/5))+1):
            inner_sum += binom(2*k1,k1+5*z)/binom(2*(k-k1),k-k1)
        for k2 in range(k-k1+1):
            result += multinomial([k1,k2,k-k1-k2])**2*(1+2*inner_sum)
    return int(np.round(result))

def moment_N_50_2(k):
    result = 0
    for k1 in range(k+1):
        inner_sum=0
        for z in range(1,int(np.floor(k1/5))+1):
            inner_sum += binom(2*k1-5*z,k1)*binom(k,k1-5*z)/binom(k,k1)/binom(2*(k-k1),k-k1)
        for k2 in range(k-k1+1):
            result += multinomial([k1,k2,k-k1-k2])**2*(1+2*inner_sum)
    return result

def moment_N_50_3(k):
    result = 0
    for k1 in range(k+1):
        temp = f(k1,k)
        for k2 in range(k-k1+1):
            result += multinomial([k1,k2,k-k1-k2])**2*(1+temp)
    return result

def moment_N_50_tilde(k):
    return 1/9**k*moment_N_50_3(k)


def A_test(A):
    boo = True
    n = 2*(A.shape[0]-2)
    if np.sum(A)!=3*n:
        boo = False
        #print('Test 1 failed')
    if np.sum(np.sum(A,axis=0)==5)!=12:
        boo = False
        #print('Test 2 failed')
    if np.sum(np.sum(A,axis=0)==6)!=int(n/2-10):
        boo = False
        #print('Test 3 failed')
    return boo
	
def spiral_to_A(n,penta_indexes,to_print=False):
    m = int(n/2+2)
    A = np.zeros((m,m))

    # Create a counter for each vertex how many connenctions are left
    free_edge_counter = np.zeros(m)
    for i in range(m):
        if np.sum(i==penta_indexes)==1:
            free_edge_counter[i] = 5
        else:
            free_edge_counter[i] = 6
    ####print(Free_edge_counter)
    #Spiral conjecture statement (1):
    # Each new face in the spiral after the second shares an edge with both its immediate predecessor in the spiral and ....
    for i in range(m-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
    free_edge_counter[0] = free_edge_counter[0]-1
    free_edge_counter[-1] = free_edge_counter[-1]-1
    free_edge_counter[1:-1] = free_edge_counter[1:-1]-2
    ####print(Free_edge_counter)


    #Spiral conjecture statement (2):
    #the first face first_open_face in the preceding spiral that still has an open edge
    new_face = 0
    open_face = 2
    left_overs_start = False
    cuts = np.ones((m,2))*m
    cuts[:,1] = np.arange(0,m)
    cuts_counter = 0
    while np.sum(free_edge_counter)>2 and not left_overs_start:

        # Basic step
        if free_edge_counter[new_face]>0 and free_edge_counter[open_face]>0 and A[new_face,open_face]==0:
            #if to_print:
                #print(new_face,open_face)
            A[new_face,open_face] = 1
            A[open_face,new_face] = 1
            free_edge_counter[new_face] -= 1
            free_edge_counter[open_face] -= 1
            open_face += 1
        #Case: 'Cut corner' 
        #new_face is complete => Check all later faces whether they are complete, and if so, 
        #connected their precursor and successor
        if free_edge_counter[new_face] == 0:
            if new_face < m-1:
                for i in range(new_face+1,m-2):
                    if free_edge_counter[i]==0 and len(np.where(free_edge_counter[0:i]>0)[0])>0 and len(np.where(free_edge_counter[i:m]>0)[0])>0:
                        # Find largest precursor and smallest successor who are both not complete:
                        #print(i)
                        max_pre = np.max(np.where(free_edge_counter[0:i]>0)[0])
                        min_suc = i + np.min(np.where(free_edge_counter[i:m]>0)[0])
                        if A[max_pre,min_suc]==0:
                            #if to_print:
                            #    print('Case: Cut corner')
                            #    print(max_pre,min_suc)
                            cuts[min_suc,:] = int(np.min([max_pre,cuts[min_suc,0]])),int(min_suc)
                            cuts_counter += 1
                            #if to_print:
                            #    print('----------------')
                            A[max_pre,min_suc] = 1
                            A[min_suc,max_pre] = 1
                            free_edge_counter[max_pre] -= 1
                            free_edge_counter[min_suc] -= 1
                            if np.sum(free_edge_counter)<3:
                                left_overs_start = True
            if not left_overs_start:
                new_face = np.min(np.where(free_edge_counter>0)[0])
                if cuts_counter > 0 and cuts[new_face,0] < m:
                    open_face = np.max(np.where(A[int(cuts[new_face,0]),:]==1)[0])
                else:
                    open_face = np.max(np.where(A[new_face-1,:]==1)[0])
    
    left_overs = np.where(free_edge_counter)[0]
    if len(left_overs)==2:
    #    if to_print:
    #        print(left_overs[0],left_overs[1])
        A[left_overs[0],left_overs[1]] = 1
        A[left_overs[1],left_overs[0]] = 1
        free_edge_counter[left_overs[0]] -= 1
        free_edge_counter[left_overs[1]] -= 1
    #if to_print:
    #    print(A_test(A))
    return A

def character(A,alpha,beta):
    D = np.diag(np.sum(A,axis=0))
    return np.trace(expm(alpha*A+beta*D))

def adjacencyTn6(A):
    hexagon_indices = np.where(np.sum(A,axis=0)==6)[0]
    A6 = A[hexagon_indices,:]
    A6 = A6[:,hexagon_indices]
    return A6
    
    
    
######################## Pentagon cluster analysis #######################################
### Create a readible and comprehensed version of all adjacency matrices of C_n
def run_bucky_planarread(n, dual = True, readable = False):
    
    #Check whether unreadible file already exists
    if os.path.exists(path_database + "/C" + str(n) + "_dual.gz"):
        print("Requested file does already exist in " + path_database)
    else:
        #Create unreadible file using buckygen
        os.system(path_buckygen + " " + str(int(n/2 + 2)) + " " + path_database + "/C" + str(n) + "_dual >/dev/null 2>&1");
        
        #Gzip the unreadible file
        os.system("cd " + path_database + "; gzip C" + str(n) + "_dual");
        print(str(n) + " done.")
    
    #In case the readible file needs to remain
    if readable:
        #Check whether readile file already exists
        if os.path.exists(path_database + "/C" + str(n) + "_dual_read"):
            print("Requested file does already exist in " + path_database)
        else:
            #Gunzip the file
            os.system("cd " + path_database + "; gunzip C" + str(n) + "_dual.gz")
            
            #Apply planarread to it
            os.system(path_planarread + " <" + path_database + "/C" + str(n) + "_dual" + 
                      ">" + path_database + "/C" + str(n) + "_dual_read")
            #Gzip it back
            os.system("cd " + path_database + "; gzip C" + str(n) + "_dual");
    return

def import_Cn(n):
    m = int(n / 2 + 2)
    #If the readible file does not exist yet, run the planarread algorithm
    if not os.path.exists(path_database + "/C" + str(n) + "_dual_read"):
        run_bucky_planarread(n,True,True)
    path = path_database + "/C" + str(n) + "_dual_read"
    
    M = np.loadtxt(open(path,'rt').readlines()[:-1],usecols = (0,1,2,3,4,5))
    iso = int(M.shape[0]/m)
    A_tensor = np.zeros((m,m,iso))
    for i in range(iso):
        A = M[i*m:(i+1)*m,:]
        for j in range(m):
            for col in range(6):
                if A[j,col] != 0:
                    A_tensor[j,int(A[j,col]-1),i] = 1
    #Delete the readable file
    os.system("rm " + path_database + "/C" + str(n) + "_dual_read")
    return A_tensor

def pentagon_cluster(A):
    # Input must be a two--dimensional matrix, not a tensor
    m = A.shape[0]
    pentagon_indices = np.where(np.sum(A,axis = 0) == 5)[0]
    A5 = A[pentagon_indices,:]
    A5 = A5[:,pentagon_indices]
    A5 = nx.Graph(A5)
    return [len(c) for c in sorted(nx.connected_components(A5), key=len, reverse=True)]

###Compute all partitions of a given number n
def partitions_yield(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield tuple(a[: k + 2])
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield tuple(a[: k + 1])

def partitions(p):
    temp = list(set(partitions_yield(p)))
    num = len(temp)
    result = np.zeros((num,p))
    for i in range(num):
        temp2 = np.sort(temp[i])[::-1]
        result[i,0:len(temp2)] = temp2
    return result[result[:, 0].argsort()]
    
###Compute the sizes of all pentagonal clusters for a given (dual) fullerene represented by adjacency matrix A
def pentagon_cluster(A):
    # Input must be a two--dimensional matrix, not a tensor
    m = A.shape[0]
    pentagon_indices = np.where(np.sum(A,axis = 0) == 5)[0]
    A5 = A[pentagon_indices,:]
    A5 = A5[:,pentagon_indices]
    A5 = nx.Graph(A5)
    result = np.zeros(12)
    temp = np.sort([len(c) for c in sorted(nx.connected_components(A5), key=len, reverse=True)])[::-1]
    result[0:len(temp)] = temp
    return result
    
###For a given partition and a partition table, find the table's row in which the partition can be found
def find_partition(part,part_table):
    n = len(part)
    return np.where(np.sum(part_table[:,0:n]==part,axis=1)==n)[0][0]

###Compute the frequency of all partitions of 12 occuring in Cn
def partition_freq(n):
    pt = partitions(12)
    result = np.zeros(pt.shape[0])
    A_tensor = import_Cn(n)
    iso = A_tensor.shape[-1]
    for i in range(iso):
        result[find_partition(pentagon_cluster(A_tensor[:,:,i]),pt)] += 1
    return result
