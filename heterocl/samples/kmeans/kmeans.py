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

def top(target = None):
  #X is a placeholder composed of all the points. The last column is added to indicate the category of the point.
  X = hcl.placeholder((N, D+1))
  #centers is a placeholder composed of the clustering centers. The last column indicates the serial number of the category.
  centers = hcl.placeholder((K, D))

  def kmeans(X, centers):
    with hcl.stage("S") as S:
      with hcl.for_(0, Loop):
        with hcl.for_(0, N) as n:	#for each point, calculate the distance between it and each center. Choose the closest center as its category and write in the last column.
          mindis = hcl.local(100000)
          with hcl.for_(0, K) as k:
            temp = hcl.local(0)
            with hcl.for_(0, D) as d:
              temp[0] += (X[n, d]-centers[k, d]) * (X[n, d]-centers[k, d])
            with hcl.if_(temp[0] < mindis[0]):
              mindis[0] = temp[0]
              X[n, D] = k
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

    return S

  o = hcl.make_schedule([X, centers], kmeans)
  #o[kmeans.S.U].compute_at(o[kmeans.S.A], kmeans.S.A.axis[0])
  print hcl.lower(o, [X, centers])
  return hcl.build(o, [X, centers], target = target)

f = top()
X0 = np.random.randint(100, size = (N, D+1))
center = random.sample(range(N),K)
centers0 = X0[center,:-1] # Choose some points of the all randomly as the initial centers

hcl_X = hcl.asarray(X0, dtype = hcl.Int())
hcl_centers = hcl.asarray(centers0, dtype = hcl.Int())
f(hcl_X, hcl_centers)

from kmeans_golden import kmeans_golden

kmeans_golden(Loop, K, N, D, X0, centers0)

assert np.allclose(hcl_X.asnumpy(), X0)
assert np.allclose(hcl_centers.asnumpy(), centers0)

"""
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
"""




