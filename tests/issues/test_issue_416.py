# -*- coding: utf-8 -*-
import heterocl as hcl
import time
import numpy as np
import math

N = 7 * 7
max_bit = int(math.ceil(math.log(N, 2)))
data_size = (10, 1800)
dtype_image = hcl.UInt(N)
dtype_knnmat = hcl.UInt(max_bit)
hcl.init(dtype_image)

def top(target=None):

    # Algorithm definition (§1)
    def knn(test_image, train_images):

        # Imperative programming and bit operations (§2)
        def popcount(num):
            out = hcl.scalar(0, "out")
            with hcl.for_(0, train_images.type.bits) as i:
                # Bit selection operation
                out.v += num[i]
            return out.v

        # This function update the candidates, i.e., `knn_mat`. Here we mutate
        # through the shape of tensor `dist`. For each `dist` value, if it is
        # smaller than the maximum candidate, we replace it.
        def update_knn(dist, knn_mat, i, j):
            max_id = hcl.scalar(0, "max_id")
            with hcl.for_(0, 3) as k:
                with hcl.if_(knn_mat[i][k] > knn_mat[i][max_id.v]):
                    max_id.v = k
            with hcl.if_(dist[i][j] < knn_mat[i][max_id.v]):
                knn_mat[i][max_id.v] = dist[i][j]

        # Main algorithm (§3)
        # Fist step: XOR (§3.1)
        diff = hcl.compute(train_images.shape,
                           lambda x, y: train_images[x][y] ^ test_image,
                           "diff")

        # Second step: popcount (§3.2)
        dist = hcl.compute(diff.shape,
                           lambda x, y: popcount(diff[x][y]),
                           "dist")


        # Third step: initialize the candidates (§3.3)
        knn_mat = hcl.compute((10, 3), lambda x, y: 50, "knn_mat")


        # Fourth step: update the candidates (§3.4)
        hcl.mutate(dist.shape,
                        lambda x, y: update_knn(dist, knn_mat, x, y),
                        "knn_update")

        # Final step: return the candidates (§3.5)
        return knn_mat

    # Inputs/Outputs definition (§4)
    # Scalars (§4.1)
    test_image = hcl.placeholder((), "test_image")
    # Tensors (§4.2)
    train_images = hcl.placeholder(data_size, "train_images")

    # Data type customization (§5.1)
    scheme = hcl.create_scheme([test_image, train_images], knn)
    scheme.downsize([knn.dist, knn.dist.out, knn.knn_mat], dtype_knnmat)

    # Compute customization (§5.2)
    s = hcl.create_schedule_from_scheme(scheme)

    diff = knn.diff
    dist = knn.dist
    knn_update = knn.knn_update

    # Merge loop nests
    s[diff].compute_at(s[dist], dist.axis[1])
    s[dist].compute_at(s[knn_update], knn_update.axis[1])

    # Reorder loop to expose more parallelism
    s[knn_update].reorder(knn_update.axis[1], knn_update.axis[0])

    s[knn.knn_mat].unroll(0)
    s[knn.knn_mat].unroll(1)
    # Pipeline the outer loop and let the inner loop unrolled automatically
    s[knn_update].pipeline(knn_update.axis[1])

    s.partition(train_images, dim=1)
    s.partition(knn.knn_mat)

    # Move data to and from device
    if isinstance(target, hcl.Platform):
        s.to(train_images, target.xcel, burst_len=16)
        s.to(knn_update.knn_mat, target.host, burst_len=16)

    # At the end, we build the whole offloaded function.
    return hcl.build(s, target=target)


def test_code_gen_knn():
    # Generate HLS kernel code and OpenCL host code
    hcl.init(dtype_image)
    target = hcl.Platform.aws_f1
    target.config(compiler="vitis", backend="vhls", mode="debug")
    code = top(target)
    assert "buffer_test_image = test_image" in code, code


if __name__ == "__main__":
    test_code_gen_knn()
