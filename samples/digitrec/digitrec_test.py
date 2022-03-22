# -*- coding: utf-8 -*-
import heterocl as hcl
import time
import numpy as np
import math
from digitrec_data import read_digitrec_data
import sys

N = 7 * 7
max_bit = int(math.ceil(math.log(N, 2)))
data_size = (10, 1800)

dtype_image = hcl.UInt(N)
dtype_knnmat = hcl.UInt(max_bit)
hcl.init(dtype_image)

num = int(sys.argv[1])

def top(target=None):

    def knn(test_image, train_images):
        def popcount(x,y,num):
            out = hcl.scalar(0, "out")
            with hcl.for_(0, train_images.dtype.bits) as i:
                out.v += num[x][y][i]
            return out.v

        def update_knn(dist, knn_mat, i, j):
            max_id = hcl.scalar(0, "max_id")
            with hcl.for_(0, 3) as k:
                with hcl.if_(knn_mat[i][k] > knn_mat[i][max_id.v]):
                    max_id.v = k
            with hcl.if_(dist[i][j] < knn_mat[i][max_id.v]):
                knn_mat[i][max_id.v] = dist[i][j]

        diff = hcl.compute(train_images.shape,
                           lambda x, y: train_images[x][y] ^ test_image[0],
                           "diff")

        dist = hcl.compute(diff.shape,
                           lambda x, y: popcount(x,y,diff),
                           "dist")


        knn_mat = hcl.compute((10, 3), lambda x, y: 50, "knn_mat")
        hcl.mutate(dist.shape,
                        lambda x, y: update_knn(dist, knn_mat, x, y),
                        "knn_update")

        return knn_mat

    
    # run #num kernels on input, and reduce finally
    def many_kernels(input_image, *train_images_arr):
        results = list()
        for i in range(num):
            res_mat = knn(input_image, train_images_arr[i])
            results.append(res_mat)
        
        # average the 10x3 matrices
        def avg(x, y):
            final = hcl.scalar(0, "value")
            for i in range(num):
                final.v += train_images_arr[i][x, y]
            return final.v
        return hcl.compute((10,3), lambda x, y: avg(x,y), name="final")


    test_image = hcl.placeholder((1,), "test_image")    
    arr = list()
    for i in range(num):
        train_images = hcl.placeholder(data_size, "train_images" + str(i))
        arr.append(train_images)

    args = [test_image] + arr
    s = hcl.create_schedule(args, many_kernels)

    # Move data to and from device
    if isinstance(target, hcl.Platform):
        s.to(train_images, target.xcel, burst_len=16)
        s.to(knn_update.knn_mat, target.host, burst_len=16)

    # At the end, we build the whole offloaded function.
    # print(hcl.lower(s))

    start = time.time()
    f = hcl.build(s, target=target)
    total_time = (time.time() - start)
    return f, total_time


offload, build_time = top("vhls")

def knn_vote(knn_mat):
    knn_mat.sort(axis = 1)
    knn_score = np.zeros(10)

    for i in range(0, 3):
        min_id = np.argmin(knn_mat, axis = 0)[i]
        knn_score[min_id] += 1

    return np.argmax(knn_score)


if __name__ == "__main__":

    train_images, _, test_images, test_labels = read_digitrec_data()
    correct = 0.0

    total_time = 0
    hcl_train_images_list = list()
    for j in range(num):
        hcl_train_images = hcl.asarray(train_images, dtype_image)
        hcl_train_images_list.append(hcl_train_images)

    hcl_knn_mat = hcl.asarray(np.zeros((10, 3)), dtype_knnmat)
    # start = time.time()
    # offload(test_images[0], *hcl_train_images_list, hcl_knn_mat)
    # total_time = total_time + (time.time() - start)
    total_time = 0.

    # Convert back to a numpy array
    knn_mat = hcl_knn_mat.asnumpy()

    print("[build, execution] {:.4f} {:.4f}".format(build_time, total_time))

    with open("time.txt", "a+") as fp:
        fp.write(f"{num},{build_time},{total_time}\n")