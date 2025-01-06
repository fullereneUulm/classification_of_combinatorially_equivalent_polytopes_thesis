import numpy as np
import os
import matplotlib.pyplot as plt

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

exp=np.zeros((6,7))
var=np.zeros((6,7))
L_d=4653
d=10

s=10000
for n in range(100,110,10):
    i=int(n/10-6)
    f = int(n/2+2)
    
    #Create (3-regular) matrix of C_n via buckygen
    str1 = "./buckygen "+ str(f) +" -d C" +str(n);
    os.system(str1)
    
    #Make it readleable via planarread
    str1 = "./planarread <C"+ str(n) +" >C"+str(n)+"_read";
    os.system(str1)
    
    #import the created file
    str1 = "C"+str(n)+"_read"
    x=np.loadtxt(str1, delimiter = "  ", usecols = (0,1,2), dtype = int)
    xshape=x.shape
    num_iso=int(xshape[0]/n)   
    
    #Generate random number in [1,num_iso]
    random_i=np.random.randint(low=0, high=num_iso-1, size=s)
    
    #Create a matrix with all Newton polynomials 
    realization=np.zeros(s)
    for j in range(0,s):
        A=create_Adj(x[random_i[j]*n:(random_i[j]+1)*n,:], n)
        realization[j]=NP_k(A,d)-n*L_d
    exp[d-5,i]=np.mean(realization)
    var[d-5,i]=np.var(realization)  
    
        
        
        
    str1="rm C"+str(n)
    os.system(str1)
    str1="rm C"+str(n)+"_read"
    os.system(str1)
    
_ = plt.hist(realization, bins='auto')
print("Done")
