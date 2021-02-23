import heterocl as hcl
import numpy as np
import math
import time
import os

hcl.init()

F = 3
FILTER_SIZE = F * F
PADDING = F - 1
MAX_FMAP = 5184
O_WIDTH = 4
N_CHANNEL1=16
N_CHANNEL2 = 32
I_WIDTH1 = 16
I_WIDTH2 = 8
FC1_UNITS = O_WIDTH*O_WIDTH*N_CHANNEL2
FC2_UNITS = 256
BUS_WIDTH = 32
MAX_W_CONV = 4608
OUT = 10
MAX_W_FC = FC1_UNITS*FC2_UNITS
TEST_SIZE = 100


def pad(input, M, I):

    ifmap_size = hcl.scalar(I*I)
    ofmap_size = hcl.scalar((I+PADDING)*(I+PADDING))

    output = hcl.compute((MAX_FMAP,), lambda x: 0)

    i_index = hcl.scalar(0)
    o_index = hcl.scalar(0)

    with hcl.Stage("padstage"):
        with hcl.for_(0,M, name = "padm") as m:
            with hcl.for_(0,I, name = "padx") as x:
                with hcl.for_(0,I, name = "pady") as y:
                    i_index.v = x + y*I + m*ifmap_size
                    o_index.v = (x + PADDING/2) + (y + PADDING/2)*(I + PADDING) + m*ofmap_size

                    output[o_index.v] = input[i_index.v]
                    
    return output

def if_mac(x, y, I):
    #with hcl.if_(hcl.or_((x < PADDING / 2), x >= (I - PADDING / 2),(y < PADDING / 2) ,(y >= (I - PADDING / 2)))):
    output = hcl.scalar(0)

    with hcl.if_(x < PADDING / 2):
        output.v = 0
    with hcl.elif_(x >= (I - PADDING / 2)):
        output.v = 0
    with hcl.elif_(y < PADDING / 2):
        output.v = 0
    with hcl.elif_(y >= (I - PADDING / 2)):
        output.v = 0
    with hcl.else_():
        output.v = 1
    return output.v

def conv(Fmap, Filter, threshold, M ,N ,I):
    O = hcl.scalar(I-F+1)
    ifmap_size = hcl.scalar(I*I)
    ofmap_size = hcl.scalar(O*O)
    sum = hcl.scalar(0)
    o_index = hcl.scalar(0)
    cum_one_out = hcl.scalar(0)
    one_out = hcl.scalar(0)
    mac_num = hcl.scalar(0)
    i_index = hcl.scalar(0)
    w_index = hcl.scalar(0)  
    output = hcl.compute((MAX_FMAP,), lambda x: 0)

    with hcl.Stage("convstage"):
        with hcl.for_(0,N, name = "convn") as n:
            with hcl.for_(0,O, name = "convx") as x:
                with hcl.for_(0,O, name = "convy") as y:
                    sum.v = 0
                    o_index.v = x + y * O + n * ofmap_size
                    cum_one_out.v = 0
                    with hcl.for_(0,M, name = "convm") as m:
                        one_out.v = 0
                        mac_num.v = 0

                        with hcl.for_(0,F, name = "convc") as c:
                            with hcl.for_(0,F, name = "convr") as r:
                                with hcl.if_(if_mac(x + c, y + r, I) == 1):
                                    i_index.v = x + c + (y + r) * I + m * ifmap_size
                                    w_index.v = c + r * F + (n + m * N) * FILTER_SIZE
                                    one_out.v += (Fmap[i_index.v] == Filter[w_index.v])
                                    mac_num.v += 1
                        cum_one_out.v += one_out[0]
                        sum.v += (one_out[0] << 1) - mac_num[0]
                    with hcl.if_(sum.v > threshold[o_index.v]):
                        output[o_index.v] = 1 
                    with hcl.else_(): 
                        output[o_index.v] = 0
    return output


def popcount(M, x):
    count = hcl.scalar(0)
    with hcl.for_(0,M) as i:
        count.v += x[i]

    return count.v


def packed_conv(Fmap, Filter, threshold, M ,N ,I):

    O = hcl.scalar(I-F+1)
    ifmap_size = hcl.scalar(I*I)
    ofmap_size = hcl.scalar(O*O)

    output = hcl.compute((MAX_FMAP,), lambda x: 0)

    new_kernels = hcl.compute ((N,F,F), lambda a,b,c: 0, dtype = hcl.UInt(64))
    new_fmaps = hcl.compute ((I,I), lambda a,b: 0, dtype = hcl.UInt(64))

    with hcl.Stage("packedstage1"):
        with hcl.for_(0,I, name = "packedy1") as y:
            with hcl.for_(0,I, name = "packedx1") as x:
                with hcl.for_(0,M, name = "packedm1") as m:
                    new_fmaps[y][x][m] = Fmap[x + y * I + m * ifmap_size]

    with hcl.Stage("packedstage2"):
        with hcl.for_(0,N, name = "packedn1") as n:
            with hcl.for_(0,F, name = "packedc1") as c:
                with hcl.for_(0,F, name = "packedr1") as r:
                    with hcl.for_(0,M) as m:
                        new_kernels[n][c][r][m] = Filter[c + r * F + (n + m * N) * FILTER_SIZE]


    sum = hcl.scalar(0)
    o_index = hcl.scalar(0)
    one_out = hcl.scalar(0)
    mac_num = hcl.scalar(0)
    temp = hcl.scalar(0,dtype = hcl.UInt(64))

    with hcl.Stage("packedstage3"):
        with hcl.for_(0,N, name = "packedn2") as n:
            with hcl.for_(0,O, name = "packedx2") as x:
                with hcl.for_(0,O, name = "packedy2") as y:
                    sum.v = 0
                    o_index.v = x + y * O + n * ofmap_size
                    one_out.v = 0
                    mac_num.v = 0
                    temp.v = 0 

                    with hcl.for_(0,F, name = "packedc2") as c:
                        with hcl.for_(0,F, name = "packedr2") as r:
                            with hcl.if_(if_mac(x+c,y+r,I) == 1):
                                temp.v = ~(new_fmaps[y+r][x+c] ^ new_kernels[n][c][r])
                                one_out.v += popcount(M,temp.v)
                                mac_num.v += 1

                    sum.v = (one_out[0] << 1) - mac_num[0]*M
                    with hcl.if_(sum.v > threshold[o_index.v]):
                        output[o_index.v] = 1 
                    with hcl.else_(): 
                        output[o_index.v] = 0
    return output

def max_pool(input, M, I):
    O = hcl.scalar(I/2)
    ifmap_size = hcl.scalar(I*I)
    ofmap_size = hcl.scalar(O*O)

    output = hcl.compute((MAX_FMAP,), lambda x: 0)
    i_index = hcl.scalar(0)
    o_index = hcl.scalar(0)

    with hcl.Stage("maxstage"):
        with hcl.for_(0,M, name = "maxm") as m:
            with hcl.for_(0,O, name = "maxx") as x:
                with hcl.for_(0,O, name = "maxy") as y:
                    o_index.v = x + y * O + m * ofmap_size
                    max = hcl.scalar(0)
                    with hcl.for_(0,2, name = "maxc") as c:
                        with hcl.for_(0,2, name = "maxr") as r:
                            i_index.v = 2 * x + c + (2 * y + r) * I + m * ifmap_size
                            with hcl.if_(input[i_index.v] == 1):
                                #hcl.print(i_index.v)
                                max.v = 1
                    output[o_index.v] = max.v

    return output

def reshape(input):
    output = hcl.compute((MAX_FMAP,), lambda x: 0)
    o_index = hcl.scalar(0)
    i_index = hcl.scalar(0)
    with hcl.Stage("restage"):
        with hcl.for_(0, N_CHANNEL2, name = "rec") as c:
            with hcl.for_(0, O_WIDTH, name = "rey") as y:
                with hcl.for_(0, O_WIDTH, name = "rex") as x:
                    o_index.v = c + (x + y * O_WIDTH ) * N_CHANNEL2
                    i_index.v = x + y * O_WIDTH + c * O_WIDTH*O_WIDTH
                    output[o_index.v] = input[i_index.v]
    return output

def dense(input, weight, bias, M, N, use_relu,c):
    max = hcl.scalar(-100, dtype = hcl.Int())
    output = hcl.compute((N,), lambda x: 0) # this N causes error for some reason; hard code in a number for now
    plus = hcl.scalar(0, dtype = hcl.Float())
    one_out = hcl.scalar(0)
    w_index = hcl.scalar(0)
    biased = hcl.scalar(0)

    with hcl.Stage("densestage"):
        with hcl.for_(0,N, name = "densen") as n:
            one_out.v = 0
            with hcl.for_(0,M, name = "densem") as m:
                w_index.v = m*N+n
                # hcl.print(w_index)
                one_out.v += (input[m] == weight[w_index.v])

            # FIXME: it works with c[0] but not c
            one_out.v = (2*one_out.v-M)*c[0]
            biased.v = one_out.v + bias[n]
            with hcl.if_(use_relu == 1):
                with hcl.if_(biased.v>0):
                    output[n] = 1
                with hcl.else_():
                    output[n] = 0
            with hcl.else_():
                with hcl.if_(biased.v>max.v):
                    max.v = biased.v
                    output[n] = 1
                with hcl.else_():
                    output[n] = 0
    return output

def top(Fmap,Filter1, Filter2,threshold1, threshold2,c1, c2, w_fc1, w_fc2, b_fc1, b_fc2):

    mem_conv1 = pad(Fmap, 1, I_WIDTH1)
    mem_conv2 = packed_conv(mem_conv1, Filter1, threshold1, 1, N_CHANNEL1, I_WIDTH1+PADDING)
    mem_conv3 = max_pool(mem_conv2, N_CHANNEL1,I_WIDTH1)

    mem_conv4 = pad(mem_conv3, N_CHANNEL1, I_WIDTH2)
    mem_conv5 = packed_conv(mem_conv4, Filter2, threshold2, N_CHANNEL1,N_CHANNEL2, I_WIDTH2+PADDING)
    mem_conv6 = max_pool(mem_conv5, N_CHANNEL2, I_WIDTH2)
    mem_conv7 = reshape(mem_conv6)

    mem_conv8 = dense(mem_conv7, w_fc1, b_fc1, FC1_UNITS, FC2_UNITS, 1, c1)
    mem_conv9 = dense(mem_conv8, w_fc2, b_fc2, FC2_UNITS, 10, 0, c2)

    max_id = hcl.scalar(0, dtype = hcl.Int(32))
    with hcl.for_(0,10) as i:
        with hcl.if_(mem_conv9[i] == 1):
            max_id.v = i

    return max_id

def toptop(Fmap,Filter1, Filter2,threshold1, threshold2,c1, c2, w_fc1, w_fc2, b_fc1, b_fc2):

    c = hcl.scalar(7, dtype = hcl.UInt(64))
    topoutput = hcl.compute((TEST_SIZE,), lambda x: 0)
    with hcl.Stage("mainloop"):
        with hcl.for_(0, TEST_SIZE, name = "testloop") as tests:
            topoutput[tests] = top(Fmap[tests],Filter1, Filter2,threshold1, threshold2,c1, c2, w_fc1, w_fc2, b_fc1, b_fc2)
    return topoutput

def main():
    Fmap_p = hcl.placeholder((TEST_SIZE,256), name = "Fmap_p", dtype = hcl.Int(8))
    Filter1_p = hcl.placeholder((MAX_W_CONV,), name = "Filter1_p")
    Filter2_p = hcl.placeholder((MAX_W_CONV,), name = "Filter2_p")

    threshold1_p = hcl.placeholder((MAX_FMAP,), name = "threshold1_p", dtype = hcl.Float())
    threshold2_p = hcl.placeholder((MAX_FMAP,), name = "threshold2_p", dtype = hcl.Float())
    M_p = hcl.placeholder((1,), name = "M_p")
    N_p = hcl.placeholder((1,), name = "N_p")
    I_p = hcl.placeholder((1,), name = "I_p")
    P_p = hcl.placeholder((1,), name = "P_p")
    c1_p = hcl.placeholder((1,), dtype = hcl.Float(), name="c1_p")
    c2_p = hcl.placeholder((1,), dtype = hcl.Float(), name="c2_p")
    wfc1_p = hcl.placeholder((MAX_W_FC,), name="wfc1_p")
    wfc2_p = hcl.placeholder((FC2_UNITS*OUT,), name="wfc2_p")
    bfc1_p = hcl.placeholder((FC2_UNITS,), dtype = hcl.Float(), name="bfc1_p")
    bfc2_p = hcl.placeholder((OUT,), dtype = hcl.Float(), name="bfc2_p")

    test_images = np.zeros((TEST_SIZE,256))
    test_labels = np.zeros((TEST_SIZE,))

    inputs = [Fmap_p, Filter1_p, Filter2_p, threshold1_p, threshold2_p,
        c1_p, c2_p, wfc1_p, wfc2_p, bfc1_p, bfc2_p]
    s = hcl.create_schedule(inputs, toptop)

    if os.path.exists('test_b.dat'):
        image = open('test_b.dat', "r")
        imagearray = image.read().replace(' ','').splitlines()

        label = open('label.dat', "r")
        labelarray = label.read().replace(' ','').splitlines()

        filter1 = open('weight_0b', "r")
        filter1array = filter1.read().replace(' ','').replace('\n','').split(",")
        filter1hcl = hcl.asarray(filter1array)

        filter2 = open('weight_5b', "r")
        filter2array = filter2.read().replace(' ','').replace('\n','').split(",")
        filter2hcl = hcl.asarray(filter2array)

        threshold1 = open('batchnorm1', "r")
        threshold1array = threshold1.read().replace('\n','').split(",")
        threshold1hcl = hcl.asarray(threshold1array, dtype = hcl.Float())

        threshold2 = open('batchnorm2', "r")
        threshold2array = threshold2.read().replace('\n','').split(",")
        threshold2hcl = hcl.asarray(threshold2array, dtype = hcl.Float())

        wfc1 = open('weight_10b', "r")
        wfc1array = wfc1.read().replace(' ','').replace('\n','').split(",")
        wfc1hcl = hcl.asarray(wfc1array)

        wfc2 = open('weight_12b', "r")
        wfc2array = wfc2.read().replace(' ','').replace('\n','').split(",")
        wfc2hcl = hcl.asarray(wfc2array)

        bfc1 = open('weight_11p', "r")
        bfc1array = bfc1.read().replace('\n','').split(",")
        bfc1hcl = hcl.asarray(bfc1array, dtype = hcl.Float())

        bfc2 = open('weight_13p', "r")
        bfc2array = bfc2.read().replace('\n','').split(",")
        bfc2hcl = hcl.asarray(bfc2array, dtype = hcl.Float())

        for index in range(0,TEST_SIZE):
            test_labels[index] = labelarray[index]
            for pixel in range (0,256):
                test_images[index][pixel] = imagearray[index*256 + pixel]

        imageshcl = hcl.asarray(test_images, dtype = hcl.Int(8))

        image.close()
        label.close()
        filter1.close()
        filter2.close()
        threshold1.close()
        threshold2.close()
        wfc1.close()
        wfc2.close()
        bfc1.close()
        bfc2.close()

    # generate random data
    else:
        imageshcl = np.random.randint(256, size=(TEST_SIZE,256))
        filter1hcl = np.random.randint(8, size=(MAX_W_CONV,))
        filter2hcl = np.random.randint(8, size=(MAX_W_CONV,))

        threshold1hcl = np.random.rand(MAX_FMAP,)
        threshold2hcl = np.random.rand(MAX_FMAP,)

        temp1 = np.random.rand(1,)
        temp2 = np.random.rand(1,)

        wfc1hcl = np.random.rand(MAX_W_FC,)
        wfc2hcl = np.random.rand(FC2_UNITS*OUT,)
        bfc1hcl = np.random.rand(FC2_UNITS,)
        bfc2hcl = np.random.rand(OUT,)

    opt = False
    if opt:
        s[toptop.mainloop].unroll(toptop.mainloop.testloop)
        s[toptop.mainloop.packedstage1].unroll(toptop.mainloop.packedstage1.packedy1)
        s[toptop.mainloop.packedstage1].unroll(toptop.mainloop.packedstage1.packedx1)
        s[toptop.mainloop.packedstage1].unroll(toptop.mainloop.packedstage1.packedm1)
        s[toptop.mainloop.packedstage2].unroll(toptop.mainloop.packedstage2.packedn1)
        s[toptop.mainloop.packedstage2].unroll(toptop.mainloop.packedstage2.packedc1)
        s[toptop.mainloop.packedstage2].unroll(toptop.mainloop.packedstage2.packedr1)
        s[toptop.mainloop.packedstage3].unroll(toptop.mainloop.packedstage3.packedn2)
        s[toptop.mainloop.packedstage3].unroll(toptop.mainloop.packedstage3.packedx2)
        s[toptop.mainloop.packedstage3].unroll(toptop.mainloop.packedstage3.packedy2)
        s[toptop.mainloop.packedstage3].unroll(toptop.mainloop.packedstage3.packedc2)
        s[toptop.mainloop.packedstage3].unroll(toptop.mainloop.packedstage3.packedr2)
        s[toptop.mainloop.padstage].unroll(toptop.mainloop.padstage.padx)
        s[toptop.mainloop.padstage].unroll(toptop.mainloop.padstage.pady)
        s[toptop.mainloop.padstage].unroll(toptop.mainloop.padstage.padm)
        s[toptop.mainloop.maxstage].unroll(toptop.mainloop.maxstage.maxm)
        s[toptop.mainloop.maxstage].unroll(toptop.mainloop.maxstage.maxx)
        s[toptop.mainloop.maxstage].unroll(toptop.mainloop.maxstage.maxy)
        s[toptop.mainloop.maxstage].unroll(toptop.mainloop.maxstage.maxc)
        s[toptop.mainloop.maxstage].unroll(toptop.mainloop.maxstage.maxr)
        s[toptop.mainloop.restage].unroll(toptop.mainloop.restage.rex)
        s[toptop.mainloop.restage].unroll(toptop.mainloop.restage.rey)
        s[toptop.mainloop.restage].unroll(toptop.mainloop.restage.rec)
        s[toptop.mainloop.densestage].unroll(toptop.mainloop.densestage.densem)
        s[toptop.mainloop.densestage].unroll(toptop.mainloop.densestage.densen)

    p = hcl.Platform.zc706
    p.config(compile="vivado_hls", mode="csyn")

    print(hcl.lower(s))
    f = hcl.build(s, target=p) 
    output = hcl.asarray(np.zeros(TEST_SIZE))

    # convert to hcl supported array automatically
    temp1 = math.sqrt(2 / float(FC1_UNITS))
    temp2 = math.sqrt(2 / float(FC2_UNITS))
    f(imageshcl, filter1hcl, filter2hcl, threshold1hcl, 
        threshold2hcl, temp1, temp2, wfc1hcl, wfc2hcl, bfc1hcl, bfc2hcl, output)
    outputs = output.asnumpy()

    correct = 0
    for tests in range(0,TEST_SIZE):
        if(outputs[tests] == test_labels[tests]):
            correct = correct + 1

    print("done")
    print(outputs)
    print(test_labels)
    print(correct)
    print(time.process_time())
    report = f.report()

if __name__ == "__main__":
    main()
