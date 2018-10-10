import heterocl as hcl
import numpy as np
import math
from lut import lut as lut_
from sgd import *

DTYPE = hcl.Fixed(16, 12)
LTYPE = hcl.Int(8)
FTYPE = hcl.Fixed(32, 19)

stepsize = 60000 % (1 << 13)

np_data = hcl.cast_np(np.loadtxt("data/shuffledfeats.dat"), DTYPE)
np_label = hcl.cast_np(np.loadtxt("data/shuffledlabels.dat"), LTYPE)
np_train_data = np_data[:NUM_FEATURES * NUM_TRAINING]
np_train_label = np_label[:NUM_TRAINING]
np_test_data = np_data[NUM_FEATURES * NUM_TRAINING:]
np_test_label = np_label[NUM_TRAINING:]
np_theta = hcl.cast_np(np.zeros(NUM_FEATURES), FTYPE)
np_lut = hcl.cast_np(np.array(lut_), FTYPE)

def sigmoid(x):
  if x > 4:
    return 1
  elif x < -4:
    return 0
  else:
    return 1 / (1 + math.exp(-x))

def train(np_theta):

  for i in range(0, NUM_TRAINING):
    training_label = np_train_label[i]
    training_instance = np_train_data[i*NUM_FEATURES : (i+1)*NUM_FEATURES]

    dot = np.dot(np_theta, training_instance)
    prob = sigmoid(dot)

    gradient = training_instance * (prob - training_label)
    np_theta -= gradient * stepsize

train(np_theta)

error = 0.0
for i in range(0, NUM_TESTING):
  data = np_test_data[i*NUM_FEATURES : (i+1)*NUM_FEATURES]
  dot = np.dot(data, np_theta)
  result = 1.0 if dot > 0 else 0.0
  if result != np_test_label[i]:
    error += 1

print error/NUM_TESTING
