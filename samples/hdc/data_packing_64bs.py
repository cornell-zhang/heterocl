import numpy as np
import heterocl as hcl
import pickle


with open('isolet.pkl', 'rb') as f:
    isolet = pickle.load(f)
    
trainData, trainLabels, testData, testLabels = isolet
trainLabels = np.array(trainLabels)
testLabels = np.array(testLabels)

numClasses = 26
quantLevels = 100

in_train = np.load("pack_train.npy")
in_test = np.load("pack_test.npy")

#data-packing bitwidth
in_bw = 64
bw = 100

def kernel(in_train, trainLabels, in_test, testLabels, rdv3, epoch):
    def popcount(value):
        #Calculate the number of one in a binary number
        count = hcl.scalar(0, "count")
        numb = hcl.scalar(value, "numb", dtype=hcl.UInt(in_bw))
        with hcl.while_(numb.v > 0):
            count.v += 1
            numb.v &= numb.v - 1
        return count.v

    def learn(k, in_train, prototype, prototypeCounter):
        #Find samples that have the label k
        match = hcl.compute(in_train.shape, lambda x,y: 0,"match",dtype=hcl.UInt(in_bw))
                        #lambda x,y: hcl.select(trainLabels[x] == k, in_train[x][y], 0), "match", dtype=hcl.UInt(in_bw))
        #Record the number of these samples
        with hcl.for_(0, in_train.shape[0]) as a:
            with hcl.if_(trainLabels[a] == k):
                max[k] += 1
                with hcl.for_(0,in_train.shape[1]) as b:
                    match[a][b] = in_train[a][b]
            with hcl.else_():
                max[k] = max[k]
                with hcl.for_(0,in_train.shape[1]) as b:
                    match[a][b] = 0

        #Do hdc sum on these samples' hdv
        r = hcl.reduce_axis(0, in_train.shape[0], 'r')
        with hcl.for_(0, in_bw) as bit:
            bit_sum = hcl.compute((in_train.shape[1],), lambda y: hcl.sum(match[r][y][bit], axis=r), "result")
            #Do the binary voting
            sum1 = hcl.compute((in_train.shape[1],), lambda x: 0, "sum1", dtype=hcl.UInt(1))
            with hcl.if_(max[k] % 2 == 0):
                hcl.update(sum1, lambda x: hcl.select(bit_sum[x] + pack_rdv3[k][x][bit] - max[k]/2 > 0, 1, 0))
            with hcl.else_():
                hcl.update(sum1, lambda x: hcl.select(bit_sum[x] - max[k]/2 > 0, 1, 0))
            #Push the binary sum to prototype and the original sum to prototypeCounter
            with hcl.for_(0, in_train.shape[1]) as t:
                prototype[k][t][bit] = sum1[t]
                prototypeCounter[k][t*in_bw+bit] = bit_sum[t]

    def test_hdc_accu(proto, pack_data, labels, type):

        ###data preparation
        distance1 = hcl.compute((pack_data.shape[1],), lambda x: 0, 'distance1', dtype=hcl.UInt(in_bw))
        pre_hamming = hcl.compute((pack_data.shape[1],), lambda x: 0, "pre_hamming")
        hamming_dist1 = hcl.compute((numClasses,), lambda x: 0, "hamming_dist1")
        m1 = hcl.reduce_axis(0, pack_data.shape[1], "m1")
        correct1 = hcl.scalar(0, 'correct1')
        ###

        with hcl.for_(0, pack_data.shape[0]) as i:
            hcl.print((i),"%d suc\n")
            with hcl.for_(0, numClasses) as n:
                #Do hdc multiplication(XOR) on sample[i]'s hdv and prototype[n]'s hdv (elementwise on the high-bit data)
                hcl.update(distance1, lambda x: pack_data[i][x] ^ proto[n][x])
                #Calculate the hamming distance of the two vectors by adding 1s
                hcl.update(pre_hamming, lambda x: popcount(distance1[x]))
                hcl.print((),"sum of 1s suc")
                ###########################seg fault
                hamming_dist1[n] = hcl.sum(pre_hamming[m1], axis=m1)

            #Find the one having the least hamming distance and choose its label as the predicted label
            pred1 = hcl.scalar(0, 'pred1')
            with hcl.for_(0, hamming_dist1.shape[0]) as j:
                with hcl.if_(hamming_dist1[j] < hamming_dist1[pred1]):
                    pred1.v = j

            with hcl.if_(pred1.v == labels[i]):
                correct1.v += 1

        #Print the accuracy
        all1 = hcl.scalar(pack_data.shape[0], "all1", dtype=hcl.Float(32))
        accuracy1 = hcl.compute((1,), lambda x: correct1.v/all1.v*100, "accuracy1" , dtype=hcl.Float(32))
        with hcl.if_(type == 1):
            hcl.print((correct1, pack_data.shape[0], accuracy1[0]), "Training accu: %d/%d (%.2f%%)\n")
        with hcl.else_():
            hcl.print((correct1, pack_data.shape[0], accuracy1[0]), "Testing accu: %d/%d (%.2f%%)\n")

    def update(l, prototype, prototypeCounter, max):
        hcl.print((l+1),"%d:Use hard examples to update the prototype counters.\n")

        ###data preparation
        distance = hcl.compute((in_train.shape[1],), lambda x: 0, 'distance', dtype=hcl.UInt(in_bw))
        pre_dist = hcl.compute((in_train.shape[1],), lambda x: 0, "pre_dist")
        hamming_dist = hcl.compute((numClasses,), lambda x: 0, "hamming_dist")
        m = hcl.reduce_axis(0, in_train.shape[1], "m")
        ###

        with hcl.for_(0, in_train.shape[0]) as i:
            hcl.print((i),"%d suc\n")
            # pack_proto = hcl.pack(prototype, axis=1, dtype=hcl.UInt(in_bw), name="pack_proto") 
            with hcl.for_(0, numClasses) as n:
                #Do hdc multiplication(XOR) on sample[i]'s hdv and prototype[n]'s hdv (elementwise on the high-bit data)
                hcl.update(distance, lambda x: in_train[i][x] ^ prototype[n][x])
                #Calculate the hamming distance of the two vectors by adding 1s
                hcl.update(pre_dist, lambda x: popcount(distance[x]))
                hcl.print((),"sum of 1s suc")
                hamming_dist[n] = hcl.sum(pre_dist[m], axis=m)

            #Find the one having the least hamming distance and choose it's label as the predicted label
            pred = hcl.scalar(0, 'pred')
            with hcl.for_(0, hamming_dist.shape[0]) as j:
                with hcl.if_(hamming_dist[j] < hamming_dist[pred]):
                    pred.v = j

            #Adjust the proto vectors by adding the sample vector on its label proto hdv and substrct it on its predicted proto hdv
            with hcl.if_(pred.v != trainLabels[i]):
                max[trainLabels[i]] += 1
                max[pred] -= 1
                with hcl.for_(0, in_train.shape[1]) as m:
                    with hcl.for_(0, in_bw) as bit:
                        prototypeCounter[trainLabels[i]][m*in_bw+bit] += in_train[i][m][bit]
                        prototypeCounter[pred][m*in_bw+bit] -= in_train[i][m][bit]
                        with hcl.if_(max[trainLabels[i]] % 2 == 0):
                            with hcl.if_(prototypeCounter[trainLabels[i]][m*in_bw+bit] - max[trainLabels[i]]/2 == 0):
                                prototype[trainLabels[i]][m][bit] &= 1
                        with hcl.else_():
                            prototype[trainLabels[i]][m][bit] = hcl.select(prototypeCounter[trainLabels[i]][m*in_bw+bit] - max[trainLabels[i]]/2 > 0, 1, 0)

                        with hcl.if_(max[pred] % 2 == 0):
                            with hcl.if_(prototypeCounter[pred][m*in_bw+bit] - max[pred]/2 == 0):
                                prototype[pred][m][bit] &= 1
                        with hcl.else_():
                            prototype[pred][m][bit] = hcl.select(prototypeCounter[pred][m*in_bw+bit] - max[pred]/2 > 0, 1, 0)

        #print the accuracy
        hcl.mutate((1,), lambda x: test_hdc_accu(prototype, in_train, trainLabels, 1), 'training_update')
        hcl.mutate((1,), lambda x: test_hdc_accu(prototype, in_test, testLabels, 2), 'testing_update')

    pack_rdv3 = hcl.pack(rdv3, axis=1, dtype=hcl.UInt(in_bw), name="pack_rdv3")

    #Unpacking the encoded training data and testing data
    # hdTrainData = hcl.unpack(in_train, axis=1, dtype=hcl.UInt(1), name="hdTrainData")
    # hdTestData = hcl.unpack(in_test, axis=1, dtype=hcl.UInt(1), name="hdTestData")
    # pack_train = hcl.pack(hdTrainData, axis=1, dtype=hcl.UInt(bw), name="pack_train")
    # pack_test = hcl.pack(hdTestData, axis=1, dtype=hcl.UInt(bw), name="pack_test")

    ###learn
    hcl.print((),"Learning the prototype HDVs.\n")
    #prototype is the vector used to represent a label
    prototype = hcl.compute((numClasses, in_train.shape[1]), lambda x, y: 0, "prototype", dtype=hcl.UInt(in_bw))
    prototypeCounter = hcl.compute((numClasses, in_train.shape[1]*in_bw), lambda x, y: 0, "prototypeCounter") #Every dimension is the sum of the targeted data

    #max is the number records the added vectors, used later for binary voting
    max = hcl.compute((numClasses, ), lambda x: 0) 
    hcl.mutate((numClasses,), lambda k: learn(k, in_train, prototype, prototypeCounter),"learn")

    #Test the accuracy after learning
    hcl.mutate((1,), lambda x: test_hdc_accu(prototype, in_train, trainLabels, 1), "test_train_accu")
    hcl.mutate((1,), lambda x: test_hdc_accu(prototype, in_test, testLabels, 2), "test_test_accu")

    ###update
    hcl.mutate((epoch[0],), lambda x: update(x, prototype, prototypeCounter, max),"update")

hcl.init()

hcl_in_train = hcl.placeholder((in_train.shape),"hcl_in_train", dtype=hcl.UInt(in_bw))
hcl_trainLabels = hcl.placeholder((trainLabels.shape),"hcl_trainLabels")
hcl_in_test = hcl.placeholder((in_test.shape),"hcl_in_test", dtype=hcl.UInt(in_bw))
hcl_testLabels = hcl.placeholder((testLabels.shape),"hcl_testLabels")
hcl_rdv3 = hcl.placeholder((numClasses,in_train.shape[1]*in_bw),"hcl_rdv3")
hcl_epoch = hcl.placeholder((1,),"hcl_epoch")

s = hcl.create_schedule([hcl_in_train, hcl_trainLabels, hcl_in_test, hcl_testLabels, hcl_rdv3, hcl_epoch], kernel)

target = hcl.platform.zc706
# target.config(compile="vitis", mode="hw_exe")
# target.config(compile="vivado_hls", mode="csim|csyn")
target.config(compile="vitis", mode="hw_exe")

# print(hcl.lower(s))
f = hcl.build(s, target)



_in_train = hcl.asarray(in_train, dtype=hcl.UInt(in_bw))
_trainLabels = hcl.asarray(trainLabels)
_in_test = hcl.asarray(in_test, dtype=hcl.UInt(in_bw))
_testLabels = hcl.asarray(testLabels)
_rdv3 = hcl.asarray(np.random.choice([0,1], size=(numClasses, in_train.shape[1]*in_bw)))
_epoch = hcl.asarray([30])

f(_in_train, _trainLabels, _in_test, _testLabels, _rdv3, _epoch)
