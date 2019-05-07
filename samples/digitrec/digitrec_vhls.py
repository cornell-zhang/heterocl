from digitrec_main import *

f = top('vhls_csim')

train_images, _, test_images, test_labels = read_digitrec_data()

correct = 0.0

total_time = 0
for i in range(0, 180):

    hcl_train_images = hcl.asarray(train_images, dtype_image)
    hcl_knn_mat = hcl.asarray(np.zeros((10, 3)), dtype_knnmat)

    start = time.time()
    f(test_images[i], hcl_train_images, hcl_knn_mat)
    total_time = total_time + (time.time() - start)

    knn_mat = hcl_knn_mat.asnumpy()

    if knn_vote(knn_mat) == test_labels[i]:
        correct += 1

print("Average kernel time (s): {:.2f}".format(total_time/180))
print("Accuracy (%): {:.2f}".format(100*correct/180))
