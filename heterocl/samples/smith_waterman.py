import heterocl as hcl
import numpy as np

lenA = 10
lenB = 10

ind = hcl.var("ind")
penalty = -4

def similarity_score(a, b):
  result = hcl.local(penalty, "result")
  with hcl.if_(a == b):
    result[0] = 1
  return result[0]

def find_max(A, len_):
  max_ = hcl.local(A[0], "max")
  with hcl.for_(0, len_) as i:
    with hcl.if_(A[i] > max_[0]):
      max_[0] = A[i]
  return max_[0]

seqA = hcl.placeholder((lenA,), "seqA")
seqB = hcl.placeholder((lenB,), "seqB")

matrix = hcl.compute((lenA + 1, lenB + 1), [], lambda x, y: 0, "maxtrix")

I_i = hcl.compute(matrix.shape, [], lambda x, y: 0, "I_i")
I_j = hcl.compute(matrix.shape, [], lambda x, y: 0, "I_j")

def populate_matrix():
  trace_back = hcl.placeholder((4,), "trace_back")

  def loop_body(i, j):
    with hcl.if_(hcl.and_(i != 0, j != 0)):
      trace_back[0] = matrix[i-1, j-1] + similarity_score(seqA[i-1], seqB[j-1])
      trace_back[1] = matrix[i-1, j] + penalty
      trace_back[2] = matrix[i, j-1] + penalty
      trace_back[3] = 0
      matrix[i, j] = find_max(trace_back, 4)
      with hcl.if_(ind == 0):
        I_i[i][j] = i - 1
        I_j[i][j] = j - 1
      with hcl.elif_(ind == 1):
        I_i[i][j] = i - 1
        I_j[i][j] = j
      with hcl.elif_(ind == 2):
        I_i[i][j] = i
        I_j[i][j] = j - 1
      with hcl.else_():
        I_i[i][j] = i
        I_j[i][j] = j

  hcl.mut_compute((lenA, lenB), [matrix, I_i, I_j, seqA, seqB], lambda i, j: loop_body(i, j))

P = hcl.block([matrix, I_i, I_j, seqA, seqB], lambda: populate_matrix())

consensusA = hcl.placeholder((lenA + lenB + 2,), "consensusA")
consensusB = hcl.placeholder((lenA + lenB + 2,), "consensusB")

def trace_back():
  matrix_max = hcl.local(0, "maxtrix_max")
  i_max = hcl.local(0, "i_max")
  j_max = hcl.local(0, "j_max")
  with hcl.for_(0, lenA) as i:
    with hcl.for_(0, lenB) as j:
      with hcl.if_(matrix[i, j] > matrix_max[0]):
        matrix_max[0] = matrix[i, j]
        i_max[0] = i
        j_max[0] = j

  current_i = hcl.local(i_max[0], "current_i")
  current_j = hcl.local(j_max[0], "current_j")
  next_i = hcl.local(I_i[current_i][current_j], "next_i")
  next_j = hcl.local(I_j[current_i][current_j], "next_j")
  tick = hcl.local(0, "tick")

  with hcl.while_(hcl.and_(hcl.or_(current_i[0] != next_i[0], current_j[0] != next_j[0]),
                            next_j[0] != 0, next_i[0] != 0)):
    with hcl.if_(next_i[0] == current_i[0]):
      consensusA[tick] = 0
    with hcl.else_():
      consensusA[tick] = seqA[current_i - 1]

    with hcl.if_(next_j[0] == current_j[0]):
      consensusB[tick] = 0
    with hcl.else_():
      consensusB[tick] = seqB[current_j - 1]

    current_i[0] = next_i[0]
    current_j[0] = next_j[0]
    next_i[0] = I_i[current_i][current_j]
    next_j[0] = I_j[current_i][current_j]
    tick[0] += 1

T = hcl.block([matrix, I_i, I_j, seqA, seqB, consensusA, consensusB, P], lambda: trace_back())

s = hcl.create_schedule(T)
print hcl.lower(s, [ind, seqA, seqB, consensusA, consensusB])
f = hcl.build(s, [ind, seqA, seqB, consensusA, consensusB, matrix])

ind = 0
_seqA = hcl.asarray(np.random.randint(1, 3, size = (lenA,)))
_seqB = hcl.asarray(np.random.randint(1, 3, size = (lenB,)))
_consensusA = hcl.asarray(np.zeros(lenA + lenB + 2), hcl.Int())
_consensusB = hcl.asarray(np.zeros(lenA + lenB + 2), hcl.Int())
_matrix = hcl.asarray(np.zeros(matrix.shape), hcl.Int())

f(ind, _seqA, _seqB, _consensusA, _consensusB, _matrix)

print _seqA
print _seqB
print _consensusA
print _consensusB
print _matrix

