import numpy as np
import os

#Up to n_max

n_max=70

# Functions to create different adjacency matrices and Laplacematrices
#Last function is a shortcut for trace(A^k)
def create_A(tab,rows): 
    A=np.zeros((rows,rows),dtype = int)
    for i in range(0,rows):
        cols=tab[i,:]
        cols=cols[cols != 0]
        cols=[x-1 for x in cols]
        A[i,cols]=1
    return A
    
def create_A6(tab,rows): 
    A=np.zeros((rows,rows),dtype = int)
    for i in range(0,rows):
        cols=tab[i,:]
        cols=cols[cols != 0]
        cols=[x-1 for x in cols]
        A[i,cols]=1
    penta=np.where(np.sum(A,axis=1)==5)
    penta=penta[0]
    A=np.delete(A,penta,axis=1)
    A=np.delete(A,penta,axis=0)
    return A

def create_L(tab,rows): 
    LM=np.zeros((rows,rows),dtype = int)
    for i in range(0,rows):
        cols=tab[i,:]
        cols=cols[cols != 0]
        cols=[x-1 for x in cols]
        LM[i,cols]=1

    return np.diag(sum(LM))-LM

def create_L6(tab,rows): 
    LM6=np.zeros((rows,rows),dtype = int)
    for i in range(0,rows):
        cols=tab[i,:]
        cols=cols[cols != 0]
        cols=[x-1 for x in cols]
        LM6[i,cols]=1
    penta_indices=np.where(sum(LM6)==5)
    LM6=np.delete(LM6,penta_indices[0],axis=1)
    LM6=np.delete(LM6,penta_indices[0],axis=0)
    return np.diag(sum(LM6))-LM6

def NP_k(A,k):
    return np.trace(np.linalg.matrix_power(A,k))

#import existing tables
k_star_T=np.loadtxt("k_star_T_single", delimiter = "  ", dtype = int)  
k_star_T6=np.loadtxt("k_star_T6_single", delimiter = "  ", dtype = int)
k_star_L=np.loadtxt("k_star_L_single", delimiter = "  ", dtype = int)
k_star_L6=np.loadtxt("k_star_L6_single", delimiter = "  ", dtype = int)



for n in range(24,n_max+1,2):
    #for a given n, check whether k* was caluclated already
    # 1 = has not been computed yet
    # 0 = has been computed already
    k_star_done=[]
    if sum(k_star_T[:,0]==n)==0:
        k_star_done.append(1)
    else:
        k_star_done.append(0)
    if sum(k_star_T6[:,0]==n)==0:
        k_star_done.append(1)
    else:
        k_star_done.append(0)
    if sum(k_star_L[:,0]==n)==0:
        k_star_done.append(1)
    else:
        k_star_done.append(0)
    if sum(k_star_L6[:,0]==n)==0:
        k_star_done.append(1)
    else:
        k_star_done.append(0)
    
    
    #if any k* was not computed already generate the adjacency matrices, else skip this n
    if sum(k_star_done)>0:
        f = int(n/2+2)
        
        #Create matrix of T_n (whole dualgraph) via buckygen
        str1 = "./buckygen "+ str(f) +" C" +str(n);
        os.system(str1)
        
        #Make it readleable via planarread
        str1 = "./planarread <C"+ str(n) +" >C"+str(n)+"_read";
        os.system(str1)
        
        #Delete the unreadable format
        str1="rm C"+str(n)
        os.system(str1)
        
        #import the created file
        str1 = "C"+str(n)+"_read"
        x=np.loadtxt(str1, delimiter = "  ", usecols = (0,1,2,3,4,5), dtype = int)
        xshape=x.shape
        #Compute the number of isomers
        num_iso=int(xshape[0]/f)   
        
        # k* for T_n calculated already?
        if k_star_done[0]:
            NP=np.zeros((f,num_iso),dtype=int) 
            #Create a matrix with all Newton polynomials 
            for i in range(0,num_iso):
                A=create_A(x[i*f:(i+1)*f,:], f)
                for k in range(1,f+1):
                    NP[k-1,i]=NP_k(A,k)
            
            #Find first row in NP where all values are distinct
            boo=1
            i=0;
            k_star_n=0
            while i<f and boo:
                if len(np.unique(NP[i]))==num_iso:
                    k_star_n=i+1
                    boo=0
                i=i+1
                
            if k_star_n==0:
                k_star_n=-1
                
            k_star_T=np.vstack((k_star_T,[n,k_star_n]))
            
        #k* for T6_n calculated already?
        if k_star_done[1]:
            NP=np.zeros((f,num_iso),dtype=int) 
            #Create a matrix with all Newton polynomials 
            for i in range(0,num_iso):
                A=create_A6(x[i*f:(i+1)*f,:], f)
                for k in range(1,f+1):
                    NP[k-1,i]=NP_k(A,k)
            
            #Find first row in NP where all values are distinct
            boo=1
            i=0;
            k_star_n=0
            while i<f and boo:
                if len(np.unique(NP[i]))==num_iso:
                    k_star_n=i+1
                    boo=0
                i=i+1
                
            if k_star_n==0:
                k_star_n=-1
                
            k_star_T6=np.vstack((k_star_T6,[n,k_star_n]))   
        
        #k* for L_n calculated already?
        if k_star_done[2]:
            NP=np.zeros((f,num_iso),dtype=int) 
            #Create a matrix with all Newton polynomials 
            for i in range(0,num_iso):
                A=create_L(x[i*f:(i+1)*f,:], f)
                for k in range(1,f+1):
                    NP[k-1,i]=NP_k(A,k)
            
            #Find first row in NP where all values are distinct
            boo=1
            i=0;
            k_star_n=0
            while i<f and boo:
                if len(np.unique(NP[i]))==num_iso:
                    k_star_n=i+1
                    boo=0
                i=i+1
                
            if k_star_n==0:
                k_star_n=-1
                
            k_star_L=np.vstack((k_star_L,[n,k_star_n]))  
            
        #k* for L6_n calculated already?
        if k_star_done[3]:
            NP=np.zeros((f,num_iso),dtype=int) 
            #Create a matrix with all Newton polynomials 
            for i in range(0,num_iso):
                A=create_L6(x[i*f:(i+1)*f,:], f)
                for k in range(1,f+1):
                    NP[k-1,i]=NP_k(A,k)
            
            #Find first row in NP where all values are distinct
            boo=1
            i=0;
            k_star_n=0
            while i<f and boo:
                if len(np.unique(NP[i]))==num_iso:
                    k_star_n=i+1
                    boo=0
                i=i+1
                
            if k_star_n==0:
                k_star_n=-1
                
            k_star_L6=np.vstack((k_star_L6,[n,k_star_n])) 
            
        #delete the readable format
        str1="rm C"+str(n)+"_read"
        os.system(str1)
    
# Sort all tabels according to n (first column)    
k_star_T=k_star_T[k_star_T[:,0].argsort()]  
k_star_T6=k_star_T6[k_star_T6[:,0].argsort()]
k_star_L=k_star_L[k_star_L[:,0].argsort()]    
k_star_L6=k_star_L6[k_star_L6[:,0].argsort()]  

#Save all tables
np.savetxt("k_star_T_single",k_star_T, fmt="%i", delimiter= "  ")
np.savetxt("k_star_T6_single",k_star_T6, fmt="%i", delimiter= "  ")
np.savetxt("k_star_L_single",k_star_L, fmt="%i", delimiter= "  ")
np.savetxt("k_star_L6_single",k_star_L6, fmt="%i", delimiter= "  ")

print("Done")
