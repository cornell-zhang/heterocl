# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
import numpy as np

# -----------------------------------------------------------------------------
def unpackbits(x):
  # Input:  'x' is a hex number represented as a str, up to 'bits' bits
  # Output: binary rep. of the number represented as a numpy array of ints

  n_digits = len(x) - 2
  n_pairs = n_digits // 2

  if len(x) % 2 == 0:
    # Even number of hex n_digits
    tok = [x[2*i:2*i+2] for i in range(1,n_pairs+1)]
  else:
    # Odd number of hex n_digits, erase the 'x' in the 1st digit
    tok = [x[2*i+1:2*i+3] for i in range(0,n_pairs+1)]
    tok[0] = tok[0][1]
    n_pairs = n_pairs+1

  # Convert each token into a bit array
  tok = [np.uint8(int(t,16)) for t in tok]
  bits = [np.unpackbits(t) for t in tok]

  # Concat all bit arrays, extend len to 49
  zeros = [np.zeros(49-(2*n_pairs)*4)]
  bits = np.concatenate(zeros+bits, 0)
  assert(bits.shape[0] == 49)
  return bits

# -----------------------------------------------------------------------------
def onehot(i,N):
  arr = np.zeros(N)
  arr[i] = 1.
  return arr

# -----------------------------------------------------------------------------
def read_digitrec_file(filename, read_labels=False):
  # Returns an ndarray of shape (n,49) where n is number of samples
  f = open(filename)
  digits_lst = []
  labels_lst = []

  for line in f:
    tok = line.split(',')
    digits = unpackbits(tok[0])
    digits_lst.append(digits)

    if read_labels:
      labels_lst.append( onehot(int(tok[1]),10) )

  if read_labels:
    return np.stack(digits_lst), np.stack(labels_lst)
  else:
    return np.stack(digits_lst)

# -----------------------------------------------------------------------------
def read_digitrec_data():
  DIR = os.path.dirname(os.path.realpath(__file__))
  DATA_DIR = os.path.join(DIR, "data")

  # Read training data
  train_digits = []
  train_labels = []
  for i in range(10):
    TRAIN_DATA = os.path.join(DATA_DIR, "training_set_"+str(i)+".dat")
    digits = read_digitrec_file(TRAIN_DATA)
    train_digits.append(digits)

    labels = np.zeros((digits.shape[0],10))
    labels[:,i] = 1.
    train_labels.append(labels)

  train_digits = np.stack(train_digits)
  train_labels = np.stack(train_labels)


  # Read test data
  TEST_DATA = os.path.join(DATA_DIR, "testing_set.dat")
  test_digits, test_labels = read_digitrec_file(TEST_DATA, read_labels=True)

  hcl_train_digits = []
  hcl_train_labels = []
  for i in range(0, 10):
    digits = []
    labels = []
    for j in range(0, 1800):
      a = np.array(train_digits[i][j]).astype("int")
      num = a.dot(1 << np.arange(a.size)[::-1])
      digits.append(num)
      label = train_labels[i][j]
      label_out = 0
      for k in range(0, 10):
        if label[k] == 1:
          label_out = k
      labels.append(label_out)
    hcl_train_digits.append(digits)
    hcl_train_labels.append(labels)
  hcl_train_digits = np.array(hcl_train_digits).astype("int64")
  hcl_train_labels = np.array(hcl_train_labels)

  hcl_test_digits = []
  hcl_test_labels = []
  for i in range(0, 180):
    a = np.array(test_digits[i]).astype("int")
    num = a.dot(1 << np.arange(a.size)[::-1])
    hcl_test_digits.append(num)
    label = test_labels[i]
    label_out = 0
    for j in range(0, 10):
      if label[j] == 1:
        label_out = j
    hcl_test_labels.append(label_out)
  hcl_test_digits = np.array(hcl_test_digits).astype("int64")
  hcl_test_labels = np.array(hcl_test_labels)

  return hcl_train_digits, hcl_train_labels, hcl_test_digits, hcl_test_labels

# -----------------------------------------------------------------------------
if __name__ == "__main__":
  train_digits, train_labels, test_digits, test_labels = read_digitrec_data()
  print(train_digits.shape)
  print(train_labels.shape)
  print(test_digits.shape)
  print(test_labels.shape)
