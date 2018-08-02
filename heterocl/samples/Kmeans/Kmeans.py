"""
This is the K-means clustering algorithm written in Heterocl
"""
#encoding:utf-8
import numpy as np
import heterocl as hcl
import random
#Define the number of the clustering center as K, the number of points as N, the Dim of each point as Dim, the iteration tims as Loop
K = 4
N = 60
Dim = 2
Loop = 100

X = hcl.placeholder((N,Dim+1))	#X is a placeholder composed of all the points. The last column is added to indicate the category of the point.
centerArray = hcl.placeholder((K,Dim+1))	#centerArray is a placeholder composed of the clustering centers. The last column indicates the serial number of the category.

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
					
		with hcl.for_(0, K) as j:	#for each category, calculate the average coordinate of its points and define the outcome as the new center.
			num = hcl.local(0)
			sum1 = hcl.local(0)
			sum2 = hcl.local(0)
			with hcl.for_(0,N) as a:
				with hcl.if_(X[a,Dim]==j):
					num[0] = num[0] + 1
					sum1[0] = sum1[0] + X[a,0]
					sum2[0] = sum2[0] + X[a,1]
			x1 = hcl.local(0)
			x1[0] = sum1[0]/num[0]
			x2 = hcl.local(0)
			x2[0] = sum2[0]/num[0]
			centerArray[j,0] = x1[0]
			centerArray[j,1] = x2[0]
			centerArray[j,2] = j	 
			
o = hcl.create_schedule(s)
print hcl.lower(o, [X, centerArray])
f = hcl.build(o, [X, centerArray])

X0 = np.random.randint(100, size = X.shape)
center = random.sample(range(N),K) 
centerArray0 = X0[center,:] # Choose some points of the all randomly as the initial centers

hcl_X = hcl.asarray(X0, dtype = hcl.Int())
hcl_centerArray = hcl.asarray(centerArray0, dtype = hcl.Int())

f(hcl_X, hcl_centerArray)		

print hcl_X
print hcl_centerArray

				
				
				
