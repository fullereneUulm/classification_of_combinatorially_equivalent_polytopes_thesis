import numpy as np
import os
from numpy import linalg as LA

def create_Adj(tab,rows): 
    A=np.zeros((rows,rows),dtype = int)
    for i in range(0,rows):
        cols=tab[i,:]
        cols=cols[cols != 0]
        cols=[x-1 for x in cols]
        A[i,cols]=1
    return A

def NP_k(A,k):
    return np.trace(np.linalg.matrix_power(A,k))

n=200
num_iso=214127742
str1 = "C"+str(n)+"_read"
x=np.loadtxt(str1, delimiter = "  ", usecols = (0,1,2), dtype = int)

energy=np.zeros(num_iso)

for i in range(0,num_iso):
    A=create_Adj(x[i*n:(i+1)*n-1,:], n)
    eig,_ =LA.eig(A)
    energy[i]=np.sum(np.abs(eig))
    

