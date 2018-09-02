import heterocl as hcl
import numpy as np
from sgd import *
from lut import lut as lut_

DTYPE = hcl.Fixed(16, 12)
LTYPE = hcl.Int(8)
FTYPE = hcl.Fixed(32, 19)

D_VECTOR_SIZE = 64 / DTYPE.bits
L_VECTOR_SIZE = 32 / LTYPE.bits
F_VECTOR_SIZE = 64 / FTYPE.bits

data = hcl.placeholder((NUM_FEATURES * NUM_TRAINING / D_VECTOR_SIZE,), "data", dtype = hcl.UInt(64))
label = hcl.placeholder((NUM_TRAINING / L_VECTOR_SIZE,), "label", dtype = hcl.UInt(32))
theta = hcl.placeholder((NUM_FEATURES / F_VECTOR_SIZE,), "theta", dtype = hcl.UInt(64))
lut = hcl.placeholder((LUT_SIZE,), "lut", dtype = FTYPE)

f = hcl.make_scheme([data, label, theta, lut], SgdLR)
f.downsize(SgdLR.label_local, LTYPE)
f.quantize(SgdLR.theta_local, FTYPE)
f.quantize(SgdLR.data_local, DTYPE)

s = hcl.make_schedule_from_scheme(f)
#print hcl.lower(s, [data, label, theta, lut])

f = hcl.build(s, f.inputs)

print "Reading data and preprocessing data ..."

np_data = hcl.cast_np(np.loadtxt("data/shuffledfeats.dat"), DTYPE)
np_label = hcl.cast_np(np.loadtxt("data/shuffledlabels.dat"), LTYPE)
np_theta = hcl.cast_np(np.zeros(NUM_FEATURES), FTYPE)
np_lut = hcl.cast_np(np.array(lut_), FTYPE)

print "Finishing prerpocessing data"

np_train_data = np_data[:NUM_FEATURES * NUM_TRAINING]
np_train_label = np_label[:NUM_TRAINING]
np_test_data = np_data[NUM_FEATURES * NUM_TRAINING:]
np_test_label = np_label[NUM_TRAINING:]

np_vdata = hcl.pack_np(np_train_data, DTYPE, hcl.UInt(64))
np_vlabel = hcl.pack_np(np_train_label, LTYPE, hcl.UInt(32))
np_vtheta = hcl.pack_np(np_theta, FTYPE, hcl.UInt(64))

hcl_data = hcl.asarray(np_vdata, dtype = hcl.UInt(64))
hcl_label = hcl.asarray(np_vlabel, dtype = hcl.UInt(32))
hcl_theta = hcl.asarray(np_vtheta, dtype = hcl.UInt(64))
hcl_lut = hcl.asarray(np_lut, dtype = FTYPE)

print "Training ..."

#for i in range(0, 5):
f(hcl_data, hcl_label, hcl_theta, hcl_lut)

np_theta = hcl.unpack_np(hcl_theta.asnumpy(), hcl.UInt(64), FTYPE)

print "Testing ... "

error = 0.0
for i in range(0, NUM_TESTING):
  data = np_test_data[i*NUM_FEATURES : (i+1)*NUM_FEATURES]
  dot = 0.0
  for j in range(0, NUM_FEATURES):
    dot += data[j] * np_theta[j]
  result = 1.0 if dot > 0 else 0.0
  if result != np_test_label[i]:
    error += 1.0
print error/NUM_TESTING

