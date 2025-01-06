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

#Import existing table
k_star=np.loadtxt("k_star_T6_single", delimiter = "  ", dtype = int)  


for n in range(62,65,2):
    if sum(k_star[:,0]==n)==0:
        row=int(n/2-10)
        f = int(n/2+2)
        
        #Create matrix via buckygen
        str1 = "./buckygen "+ str(f) +" C" +str(n);
        os.system(str1)
        
        #Make it readleable via planarread
        str1 = "./planarread <C"+ str(n) +" >C"+str(n)+"_read";
        os.system(str1)
        
        #import the created file
        str1 = "C"+str(n)+"_read"
        x=np.loadtxt(str1, delimiter = "  ", usecols = (0,1,2,3,4,5), dtype = int)
        num_iso=int(x.shape[0]/f)
        NP=np.zeros((f,num_iso),dtype=int)
    
        #Compute all Newton polynomials of T_n^6 (Hexagonal graph)
        for i in range(0,num_iso):
            A=create_Adj(x[i*f:(i+1)*f,:], f)
            penta=np.where(np.sum(A,axis=1)==5)
            penta=penta[0]
            A=np.delete(A,penta,axis=1)
            A=np.delete(A,penta,axis=0)
            for k in range(1,f+1):
                NP[k-1,i]=NP_k(A,k)
            
        boo=1
        i=0
        k_star_n=0
        while boo and i<f:   
            if len(np.unique(NP[i]))==num_iso:
                k_star_n=i+1
                boo=0
            i=i+1
            
        if k_star_n==0:
            k_star_n=-1
            
        k_star=np.vstack((k_star,[n,k_star_n]))    
            
        #Delete the adjacencymatrices 
        str1="rm C"+str(n)
        os.system(str1)
        str1="rm C"+str(n)+"_read"
        os.system(str1)
    
#Save the table with k*
k_star=k_star[k_star[:,0].argsort()]  
np.savetxt("k_star_T6_single",k_star, fmt="%i", delimiter= "  ")
print("Done")
