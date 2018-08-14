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
D = 32
Loop = 200

X = hcl.placeholder((N, D+1))	#X is a placeholder composed of all the points. The last column is added to indicate the category of the point.
centers = hcl.placeholder((K, D))	#centers is a placeholder composed of the clustering centers. The last column indicates the serial number of the category.

with hcl.stage("S") as s:
  with hcl.for_(0,Loop):
    with hcl.for_(0, N) as i:	#for each point, calculate the distance between it and each center. Choose the closest center as its category and write in the last column.
      mindis = hcl.local(100000)
      with hcl.for_(0, K) as u:
        temp = hcl.local(0)
        with hcl.for_(0, D) as m:
          temp[0] = temp[0] + ((X[i,m]-centers[u,m])*(X[i,m]-centers[u,m]))
        with hcl.if_(temp[0]<mindis[0]):
          mindis[0] = temp[0]
          X[i,D] = u
    num0 = hcl.compute((K,), lambda x: 0)
    sum0 = hcl.compute((K, D), lambda x, y: 0)
    #for each category, calculate the average coordinate of its points and define the outcome as the new center.
    def update_sum(k, n):
      with hcl.if_(X[n, D] == k):
        num0[k] = num0[k] + 1
        with hcl.for_(0, D) as d:
          sum0[k, d] += X[n, d]
    U = hcl.mut_compute((K, N), lambda k, n: update_sum(k, n), "U")
    A = hcl.update(centers, lambda k, d: sum0[k, d]/num0[k], "A")

o = hcl.create_schedule(s)
o[s.U].compute_at(o[s.A], s.A.axis[0])
print hcl.lower(o, [X, centers])
f = hcl.build(o, [X, centers])

X0 = np.random.randint(100, size = X.shape)
center = random.sample(range(N),K)
centers0 = X0[center,:-1] # Choose some points of the all randomly as the initial centers

hcl_X = hcl.asarray(X0, dtype = hcl.Int())
hcl_centers = hcl.asarray(centers0, dtype = hcl.Int())
f(hcl_X, hcl_centers)


#This is the same algorithm in Python.
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

print("------------result of Heterocl----------------")
print ("All of the points :")
print ("The last column indicates its category")
print hcl_X
print ("The center points :")
print ("The last column indicates its category")
print hcl_centers


print("------------result of Python----------------")
print ("All of the points :")
print ("The last column indicates its category")
print X0
print ("The center points :")
print ("The last column indicates its category")
print centers0






