import numpy as np
import random
import heterocl as hcl
from kmeans_main import top

K = 16
N = 320
dim = 32

f1 = top('vhls_csim')
f2 = top()
points_np = np.random.randint(100, size=(N, dim))
labels_np = np.zeros(N)
means_np = points_np[random.sample(range(N), K),:]

hcl_points1 = hcl.asarray(points_np)
hcl_means1 = hcl.asarray(means_np)
hcl_labels1 = hcl.asarray(labels_np)

hcl_points2 = hcl.asarray(points_np)
hcl_means2 = hcl.asarray(means_np)
hcl_labels2 = hcl.asarray(labels_np)

f1(hcl_points1, hcl_means1, hcl_labels1)
f2(hcl_points2, hcl_means2, hcl_labels2)

assert np.array_equal(hcl_labels1.asnumpy(), hcl_labels2.asnumpy())
