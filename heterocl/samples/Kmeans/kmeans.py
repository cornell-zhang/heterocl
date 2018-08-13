"""
This is the K-means clustering algorithm written in Heterocl
"""
#encoding:utf-8
import numpy as np
import heterocl as hcl
import random
#Define the number of the clustering center as K, the number of points as N, the Dim of each point as Dim, the iteration times as Loop
K = 4
N = 320
Dim = 32
Loop = 200

X = hcl.placeholder((N,Dim+1))	#X is a placeholder composed of all the points. The last column is added to indicate the category of the point.
centerArray = hcl.placeholder((K,Dim+1))	#centerArray is a placeholder composed of the clustering centers. The last column indicates the serial number of the category.
num0 = hcl.placeholder((K,))
sum0 = hcl.placeholder((K,Dim))

with hcl.stage() as s:
	with hcl.for_(0,Loop):  
		with hcl.for_(0, N) as i:	#for each point, calculate the distance between it and each center. Choose the closest center as its category and write in the last column.
			mindis = hcl.local(100000)
			with hcl.for_(0, K) as u:
				temp = hcl.local(0)
				with hcl.for_(0, Dim) as m:
					temp[0] = temp[0] + ((X[i,m]-centerArray[u,m])*(X[i,m]-centerArray[u,m])) 
				with hcl.if_(temp[0]<mindis[0]):
					mindis[0] = temp[0]
					X[i,Dim] = u	
		hcl.update(num0, lambda x: 0)
		hcl.update(sum0, lambda x,y: 0)			
		with hcl.for_(0, K) as j:	#for each category, calculate the average coordinate of its points and define the outcome as the new center.
			with hcl.for_(0,N) as a:
				with hcl.if_(X[a,Dim]==j):
					num0[j] = num0[j] + 1
					with hcl.for_(0,Dim) as p:
						sum0[j,p] = sum0[j,p] + X[a,p]
		with hcl.for_(0, K) as q:
			with hcl.for_(0, Dim) as t:
				centerArray[q,t] = sum0[q,t]/num0[q]
			centerArray[q,Dim] = q

o = hcl.create_schedule(s)
print hcl.lower(o, [X, centerArray,num0,sum0])
f = hcl.build(o, [X, centerArray,num0,sum0])

X0 = np.random.randint(100, size = X.shape)
center = random.sample(range(N),K) 
centerArray0 = X0[center,:] # Choose some points of the all randomly as the initial centers
num00 = np.zeros((K,))
sum00 = np.zeros((K,Dim))

hcl_X = hcl.asarray(X0, dtype = hcl.Int())
hcl_centerArray = hcl.asarray(centerArray0, dtype = hcl.Int())
hcl_num0 = hcl.asarray(num00, dtype = hcl.Int())
hcl_sum0 = hcl.asarray(sum00, dtype = hcl.Int())
f(hcl_X, hcl_centerArray,hcl_num0,hcl_sum0)		

print ("All of the points :")
print ("The last column indicates its category")
print hcl_X
print ("The center points :")
print ("The last column indicates its category")
print hcl_centerArray

				
				
				
