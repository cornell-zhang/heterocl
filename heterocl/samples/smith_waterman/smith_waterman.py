import heterocl as hcl
import numpy as np
import time

lenA = 128
lenB = 128
num = 1024
penalty = -4

dtype = hcl.UFixed(3)
mtype = hcl.Int(16)

def top(target = None):

  def batch_sw(seqAs, seqBs, outAs, outBs):
    with hcl.stage("B") as B:
      with hcl.for_(0, num, name="task") as t:
        smith_waterman(seqAs[t], seqBs[t], outAs[t], outBs[t])
    return B

  def smith_waterman(seqA, seqB, consensusA, consensusB):

    def similarity_score(a, b):
      result = hcl.local(penalty, "result")
      with hcl.if_(a == b):
        result[0] = 1
      return result[0]

    def find_max(A, len_):
      max_ = hcl.local(A[0], "max")
      act_ = hcl.local(0, "act")
      with hcl.for_(0, len_) as i:
        with hcl.if_(A[i] > max_[0]):
          max_[0] = A[i]
          act_[0] = i
      return max_[0], act_[0]

    matrix_max = hcl.local(0, "maxtrix_max")
    i_max = hcl.local(0, "i_max")
    j_max = hcl.local(0, "j_max")

    matrix = hcl.compute((lenA + 1, lenB + 1), lambda x, y: 0, "matrix")
    action = hcl.compute(matrix.shape, lambda x, y: 3, "action")

    def populate_matrix(i, j):
      trace_back = hcl.compute((4,), lambda x: 0, "trace_back")

      with hcl.if_(hcl.and_(i != 0, j != 0)):
        trace_back[0] = matrix[i-1, j-1] + similarity_score(seqA[i-1], seqB[j-1])
        trace_back[1] = matrix[i-1, j] + penalty
        trace_back[2] = matrix[i, j-1] + penalty
        trace_back[3] = 0
        matrix[i, j], action[i, j] = find_max(trace_back, 4)
        with hcl.if_(matrix[i, j] > matrix_max[0]):
          matrix_max[0] = matrix[i, j]
          i_max[0] = i
          j_max[0] = j

    P = hcl.mut_compute((lenA + 1, lenB + 1), lambda i, j: populate_matrix(i, j))

    def align(curr_i, curr_j, next_i, next_j):
      outA = hcl.local(0, "a")
      outB = hcl.local(0, "b")

      with hcl.if_(next_i[0] == curr_i[0]):
        outA[0] = 0
      with hcl.else_():
        outA[0] = seqA[curr_i[0] - 1]

      with hcl.if_(next_j[0] == curr_j[0]):
        outB[0] = 0
      with hcl.else_():
        outB[0] = seqB[curr_j[0] - 1]
      return outA[0], outB[0]

    def get_next(action, i, j):
      act_ = hcl.local(action[i][j], "act")
      next_i = hcl.local(0, "next_i")
      next_j = hcl.local(0, "next_j")
      with hcl.if_(act_[0] == 0):
        next_i[0] = i - 1
        next_j[0] = j - 1
      with hcl.elif_(act_[0] == 1):
        next_i[0] = i - 1
        next_j[0] = j
      with hcl.elif_(act_[0] == 2):
        next_i[0] = i
        next_j[0] = j - 1        
      with hcl.else_():
        next_i[0] = i
        next_j[0] = j
      return next_i[0], next_j[0]

    with hcl.stage("T") as T:
      curr_i = hcl.local(i_max[0], "curr_i")
      curr_j = hcl.local(j_max[0], "curr_j")
      next_i = hcl.local(0, "next_i")
      next_j = hcl.local(0, "next_j")
      next_i[0], next_j[0] = get_next(action, curr_i[0], curr_j[0])
      tick = hcl.local(0, "tick")

      with hcl.while_(hcl.or_(curr_i[0] != next_i[0], curr_j[0] != next_j[0])):
        consensusA[tick[0]], consensusB[tick[0]] = align(curr_i, curr_j, next_i, next_j)
        curr_i[0], curr_j[0] = next_i[0], next_j[0]
        next_i[0], next_j[0] = get_next(action, curr_i[0], curr_j[0])
        tick[0] += 1

    return T

  seqAs = hcl.placeholder((num, lenA), "seqAs", dtype)
  seqBs = hcl.placeholder((num, lenB,), "seqBs", dtype)
  outAs = hcl.placeholder((num, lenA + lenB), "outAs", dtype)
  outBs = hcl.placeholder((num, lenA + lenB), "outBs", dtype)

  scheme = hcl.make_scheme([seqAs, seqBs, outAs, outBs], batch_sw)
  scheme.downsize([batch_sw.B.matrix, batch_sw.B.action], mtype)
  s = hcl.make_schedule_from_scheme(scheme)
  # FIXME: Scheduling function crashed
  #o, p = s[batch_sw.B].split(batch_sw.B.task, factor = 32)
  #s[batch_sw.B].pipeline(o)
  #s[batch_sw.B].parallel(p)
  #print hcl.lower(s, [seqAs, seqBs, outAs, outBs])
  return hcl.build(s, [seqAs, seqBs, outAs, outBs], target = target)

_seqA = hcl.asarray(np.random.randint(1, 5, size = (num, lenA)), dtype)
_seqB = hcl.asarray(np.random.randint(1, 5, size = (num, lenB)), dtype)
_consensusA = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)
_consensusB = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)

f = top()
start = time.time()
f(_seqA, _seqB, _consensusA, _consensusB)
total_time = time.time() - start
print "Kernel time (s): {:.2f}".format(total_time)

#print _seqA.asnumpy()[0:2]
#print _seqB.asnumpy()[0:2]
#print _consensusA.asnumpy()[0:2]
#print _consensusB.asnumpy()[0:2]
