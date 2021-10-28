from digitrec_main import *

p = hcl.Platform.aws_f1
p.config(compiler="vitis", mode="csyn")
f = top(p)

train_images, _, test_images, test_labels = read_digitrec_data()
hcl_train_images = hcl.asarray(train_images, dtype_image)
hcl_knn_mat = hcl.asarray(np.zeros((10, 3)), dtype_knnmat)
start = time.time()
f(test_images[0], hcl_train_images, hcl_knn_mat)