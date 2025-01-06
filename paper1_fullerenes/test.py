import numpy as np
import os

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

n=60
row=int(n/2-10)
f = int(n/2+2)
num_iso=45
NP=np.zeros(num_iso,dtype=int)
    
#Create matrix via buckygen
str1 = "./buckygen "+ str(f) +" C" +str(n);
os.system(str1)
    
#Make it readleable via planarread
str1 = "./planarread <C"+ str(n) +" >C"+str(n)+"_read";
os.system(str1)
    
#import the created file
str1 = "C"+str(n)+"_read"
x=np.loadtxt(str1, delimiter = "  ", usecols = (0,1,2,3,4,5), dtype = int)
boo=1
k=1
while k<f+1 and boo==1:
    NP=np.zeros(num_iso,dtype=int)
    for i in range(0,num_iso):
        #k_star[0,row]=n
        A=create_Adj(x[i*f:(i+1)*f,:], f)
        penta=np.where(np.sum(A,axis=1)==5)
        penta=penta[0]
        A=np.delete(A,penta,axis=1)
        A=np.delete(A,penta,axis=0)
        NP[i]=NP_k(A,k)
    if len(np.unique(NP))==num_iso:
        #k_star[1,row]=k
        boo=0
    k=k+1
    
print(boo)

