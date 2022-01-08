"""
HeteroCL Tutorial : K-means Clustering Algorithm
================================================

**Author**: Yi-Hsiang Lai (seanlatias@github), Ziyan Feng

This is the K-means clustering algorithm written in Heterocl.
"""
import numpy as np
import heterocl as hcl
import time
import random
##############################################################################
# Define the number of the clustering means as K, the number of points as N,
# the number of dimensions as dim, and the number of iterations as niter
K = 16
N = 320
dim = 32
niter = 200

hcl.init()

##############################################################################
# Main Algorithm
# ==============
def top(target=None):
    points = hcl.placeholder((N, dim), "points")
    means = hcl.placeholder((K, dim), "means")
    labels = hcl.placeholder((N,), "labels")

    def kmeans(points, means, labels):
        def loop_kernel(labels):
            # assign cluster
            with hcl.for_(0, N, name="n") as n:
                min_dist = hcl.scalar(100000)
                new_label = hcl.scalar(labels[n])
                with hcl.for_(0, K) as k:
                    dist = hcl.scalar(0)
                    with hcl.for_(0, dim) as d:
                        dist_ = hcl.scalar(points[n, d]-means[k, d], "temp")
                        dist.v += dist_.v * dist_.v
                    with hcl.if_(dist.v < min_dist.v):
                        min_dist.v = dist.v
                        new_label[0] = k
                labels[n] = new_label
            # update mean
            num_k = hcl.compute((K,), lambda x: 0, "num_k")
            sum_k = hcl.compute((K, dim), lambda x, y: 0, "sum_k")
            def calc_sum(n):
                num_k[labels[n]] += 1
                with hcl.for_(0, dim) as d:
                    sum_k[labels[n], d] += points[n, d]
            hcl.mutate((N,), lambda n: calc_sum(n), "calc_sum")
            hcl.update(means,
                    lambda k, d: sum_k[k, d]//num_k[k], "update_mean")

        hcl.mutate((niter,), lambda _: loop_kernel(labels), "main_loop")

    # create schedule and apply compute customization
    s = hcl.create_schedule([points, means, labels], kmeans)
    main_loop = kmeans.main_loop
    update_mean = main_loop.update_mean

    s[main_loop].pipeline(main_loop.n)
    s[main_loop.num_k].unroll(0)
    s[main_loop.sum_k].unroll(1)
    s[main_loop.calc_sum].pipeline(0)
    s[main_loop.update_mean].unroll(1)

    s.partition(points, dim=2)
    s.partition(means, dim=0)
    s.partition(main_loop.sum_k, dim=0)
    s.partition(main_loop.num_k, dim=0)

    # data movement
    if isinstance(target, hcl.Platform):
        main = kmeans.main_loop
        s.to(labels, target.xcel)
        s.to([points, means], target.xcel, burst_len=16)
        s.to([main.means, main.labels], target.host, burst_len=16)

    print(hcl.lower(s))
    return hcl.build(s, target=target)

if __name__ == "__main__":

    points_np = np.random.randint(100, size=(N, dim))
    labels_np = np.zeros(N)
    means_np = points_np[random.sample(range(N), K), :]

    hcl_points = hcl.asarray(points_np, dtype=hcl.Int())
    hcl_means = hcl.asarray(means_np, dtype=hcl.Int())
    hcl_labels = hcl.asarray(labels_np)

    use_cpu_sim = False
    if use_cpu_sim:
        f = top()
        start = time.time()
        f(hcl_points, hcl_means, hcl_labels)
        total_time = time.time() - start
        print("Kernel time (s): {:.2f}".format(total_time))

        print("All points:")
        print(hcl_points)
        print("Final cluster:")
        print(hcl_labels)
        print("The means:")
        print(hcl_means)

        from kmeans_golden import kmeans_golden
        kmeans_golden(niter, K, N, dim, np.concatenate((points_np,
            np.expand_dims(labels_np, axis=1)), axis=1), means_np)
        assert np.allclose(hcl_means.asnumpy(), means_np)
    
    # generate HLS code and OpenCL host code
    else:
        target = hcl.Platform.aws_f1
        target.config(compiler="vitis", mode="debug")
        code = top(target)
        print(code)
