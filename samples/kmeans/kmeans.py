"""
This is the K-means clustering algorithm written in Heterocl
"""
#encoding:utf-8
import numpy as np
import heterocl as hcl
import time
import random
#Define the number of the clustering center as K, the number of points as N, the Dim of each point as Dim, the iteration times as Loop
K = 16
N = 320
D = 32
Loop = 200

hcl.init()

def top(target = None):
    points = hcl.placeholder((N, D))
    centers = hcl.placeholder((K, D))

    def kmeans(points, centers):
        def loop_kernel():
            labels = hcl.compute((N,), lambda x: 0, "labels")
            # For each point, calculate the distance between it and each
            # center. Choose the closest center as its category and write it
            # in the last column.
            with hcl.for_(0, N, name="N") as n:
                mindis = hcl.local(100000)
                with hcl.for_(0, K, name="K") as k:
                    temp = hcl.local(0)
                    with hcl.for_(0, D) as d:
                        temp[0] += (points[n, d]-centers[k, d]) * (points[n, d]-centers[k, d])
                    with hcl.if_(temp[0] < mindis[0]):
                        mindis[0] = temp[0]
                        labels[n] = k
            num0 = hcl.compute((K,), lambda x: 0)
            sum0 = hcl.compute((K, D), lambda x, y: 0)
            #for each category, calculate the average coordinate of its points and define the outcome as the new center.
            def update_sum(n):
                k = hcl.local(0)
                k[0] = labels[n]
                num0[k[0]] = num0[k[0]] + 1
                with hcl.for_(0, D) as d:
                    sum0[k[0], d] += points[n, d]
            hcl.mutate((N,), lambda n: update_sum(n), "U")
            hcl.update(centers, lambda k, d: sum0[k, d]/num0[k], "A")

        hcl.mutate((Loop,), lambda _: loop_kernel(), "S")

    o = hcl.create_schedule([points, centers], kmeans)
    o[kmeans.S].pipeline(kmeans.S.N)
    o[kmeans.S.U].unroll(kmeans.S.U.axis[0])
    fused_1 = o[kmeans.S.A].fuse(kmeans.S.A.axis[0], kmeans.S.A.axis[1])
    o[kmeans.S.A].unroll(fused_1, factor=K*D)
    return hcl.build(o, target=target)

f = top()

points_np = np.random.randint(100, size=(N, D))
labels_np = np.zeros(N)
centers_np = points_np[random.sample(range(N), K),:]

hcl_points = hcl.asarray(points_np, dtype=hcl.Int())
hcl_centers = hcl.asarray(centers_np, dtype=hcl.Int())

start = time.time()
f(hcl_points, hcl_centers)
total_time = time.time() - start
print "Kernel time (s): {:.2f}".format(total_time)

from kmeans_golden import kmeans_golden
kmeans_golden(Loop, K, N, D, np.concatenate((points_np,
    np.expand_dims(labels_np, axis=1)), axis=1), centers_np)
assert np.allclose(hcl_centers.asnumpy(), centers_np)

"""
print("------------result of Heterocl----------------")
print ("All of the points :")
print ("The last column indicates its category")
print hcl_X
print ("The center points :")
print ("The last column indicates its category")
print hcl_centers
"""



