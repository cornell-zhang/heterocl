import numpy as np

def kmeans_golden(Loop, K, N, D, X0, centers0):
    for d in range(Loop):
        for i in range(N):
            mindis0 = 100000
            for u in range(K):
                temp0 = 0
                for m in range(D):
                    temp0 = temp0 + ((X0[i,m]-centers0[u,m])*(X0[i,m]-centers0[u,m]))
                if (temp0 < mindis0):
                    mindis0 = temp0
                    X0[i,D] = u
        num00 = np.zeros((K,),dtype = np.int)
        sum00 = np.zeros((K,D), dtype = np.int)
        for j in range(K):
            for a in range(N):
                if (X0[a,D]==j):
                    num00[j] = num00[j] + 1
                    for p in range(D):
                        sum00[j,p] = sum00[j,p] + X0[a,p]
        for q in range(K):
            for t in range(D):
                centers0[q,t] = sum00[q,t]/num00[q]
