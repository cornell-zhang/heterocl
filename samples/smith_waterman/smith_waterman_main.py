"""
HeteroCL Tutorial : Smith-Waterman Genomic Sequencing
=====================================================

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this example, we demonstrate how to use a While loop in HeteroCL.
"""
import heterocl as hcl
import numpy as np
import time

lenA = 128
lenB = 128
num = 1024
penalty = -4

hcl.init()
dtype = hcl.UFixed(3)
mtype = hcl.Int(16)

def top(target=None):

    def smith_waterman(seqA, seqB, consA, consB):

        def similarity_score(a, b):
            return hcl.select(a == b, 1, penalty)

        def find_max(A, len_):
            max_ = hcl.scalar(A[0], "max")
            act_ = hcl.scalar(0, "act")
            with hcl.for_(0, len_) as i:
                with hcl.if_(A[i] > max_.v):
                    max_.v = A[i]
                    act_.v = i
            return max_.v, act_.v

        matrix_max = hcl.scalar(0, "maxtrix_max")
        i_max = hcl.scalar(0, "i_max")
        j_max = hcl.scalar(0, "j_max")

        matrix = hcl.compute((lenA + 1, lenB + 1), lambda x, y: 0, "matrix")
        action = hcl.compute(matrix.shape, lambda x, y: 3, "action")

        def populate_matrix(i, j):
            trace_back = hcl.compute((4,), lambda x: 0, "trace_back")

            with hcl.if_(hcl.and_(i != 0, j != 0)):
                trace_back[0] = matrix[i-1, j-1] + \
                                similarity_score(seqA[i-1], seqB[j-1])
                trace_back[1] = matrix[i-1, j] + penalty
                trace_back[2] = matrix[i, j-1] + penalty
                trace_back[3] = 0
                matrix[i, j], action[i, j] = find_max(trace_back, 4)
                with hcl.if_(matrix[i, j] > matrix_max.v):
                    matrix_max.v = matrix[i, j]
                    i_max.v = i
                    j_max.v = j

        P = hcl.mutate((lenA+1, lenB+1), lambda i, j: populate_matrix(i, j))

        def align(curr_i, curr_j, next_i, next_j):
            outA = hcl.scalar(0, "a")
            outB = hcl.scalar(0, "b")

            with hcl.if_(next_i.v == curr_i.v):
                outA.v = 0
            with hcl.else_():
                outA.v = seqA[curr_i.v - 1]

            with hcl.if_(next_j.v == curr_j.v):
                outB.v = 0
            with hcl.else_():
                outB.v = seqB[curr_j.v - 1]
            return outA.v, outB.v

        def get_next(action, i, j):
            act_ = hcl.scalar(action[i][j], "act")
            next_i = hcl.scalar(0, "next_i")
            next_j = hcl.scalar(0, "next_j")
            with hcl.if_(act_.v == 0):
                next_i.v = i - 1
                next_j.v = j - 1
            with hcl.elif_(act_.v == 1):
                next_i.v = i - 1
                next_j.v = j
            with hcl.elif_(act_.v == 2):
                next_i.v = i
                next_j.v = j - 1
            with hcl.else_():
                next_i.v = i
                next_j.v = j
            return next_i.v, next_j.v

        with hcl.Stage("T"):
            curr_i = hcl.scalar(i_max.v, "curr_i")
            curr_j = hcl.scalar(j_max.v, "curr_j")
            next_i = hcl.scalar(0, "next_i")
            next_j = hcl.scalar(0, "next_j")
            next_i.v, next_j.v = get_next(action, curr_i.v, curr_j.v)
            tick = hcl.scalar(0, "tick")

            with hcl.while_(hcl.or_(curr_i.v != next_i.v,
                                    curr_j.v != next_j.v)):
                consA[tick.v], consB[tick.v] = \
                    align(curr_i, curr_j, next_i, next_j)
                curr_i.v, curr_j.v = next_i.v, next_j.v
                next_i.v, next_j.v = get_next(action, curr_i.v, curr_j.v)
                tick.v += 1

    def batch_sw(seqAs, seqBs, outAs, outBs):
        hcl.mutate((num,),
                lambda t: smith_waterman(seqAs[t], seqBs[t], outAs[t], outBs[t]),
                "B")

    seqAs = hcl.placeholder((num, lenA), "seqAs", dtype)
    seqBs = hcl.placeholder((num, lenB,), "seqBs", dtype)
    outAs = hcl.placeholder((num, lenA+lenB), "outAs", dtype)
    outBs = hcl.placeholder((num, lenA+lenB), "outBs", dtype)

    scheme = hcl.create_scheme([seqAs, seqBs, outAs, outBs], batch_sw)
    scheme.downsize([batch_sw.B.matrix, batch_sw.B.action], mtype)
    s = hcl.create_schedule_from_scheme(scheme)
    o, p = s[batch_sw.B].split(batch_sw.B.axis[0], factor=32)
    s[batch_sw.B].pipeline(o)
    s[batch_sw.B].parallel(p)
    return hcl.build(s, target=target)

###############################################################################
# Test the algorithm with random numbers
_seqA = hcl.asarray(np.random.randint(1, 5, size=(num, lenA)), dtype)
_seqB = hcl.asarray(np.random.randint(1, 5, size=(num, lenB)), dtype)
_consA = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)
_consB = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)

f = top()
start = time.time()
f(_seqA, _seqB, _consA, _consB)
total_time = time.time() - start
print("Kernel time (s): {:.2f}".format(total_time))

###############################################################################
# Test the algorithm with simple inputs
_seqA_np = np.ones((num, lenA))
for i in range(0, 4):
    _seqA_np[0][i] = 2
_seqB_np = np.ones((num, lenB))
_seqA = hcl.asarray(_seqA_np, dtype)
_seqB = hcl.asarray(_seqB_np, dtype)
_consA = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)
_consB = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)
f(_seqA, _seqB, _consA, _consB)
_consA_np = _consA.asnumpy()
_consB_np = _consB.asnumpy()
for i in range(0, 256):
    if i < 124:
        assert _consA_np[0][i] == 1
    else:
        assert _consA_np[0][i] == 0
