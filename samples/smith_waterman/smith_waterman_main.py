"""
HeteroCL Tutorial : Smith-Waterman Genomic Sequencing
=====================================================

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this example, we demonstrate how to use a While loop in HeteroCL.
"""
import heterocl as hcl
import numpy as np
import time

#lenA = 128
lenA = 28
#lenB = 128
lenB = 28
#num = 1024
num = 64
penalty = -4

hcl.init()
dtype = hcl.UFixed(3)
mtype = hcl.Int(16)

def top(target=None):

    def smith_waterman(seqA, seqB, consA, consB):

        def similarity_score(a, b):
            return hcl.select(a == b, 1, penalty)

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
                trace_back[0] = matrix[i-1, j-1] + \
                                similarity_score(seqA[i-1], seqB[j-1])
                trace_back[1] = matrix[i-1, j] + penalty
                trace_back[2] = matrix[i, j-1] + penalty
                trace_back[3] = 0
                matrix[i, j], action[i, j] = find_max(trace_back, 4)
                with hcl.if_(matrix[i, j] > matrix_max[0]):
                    matrix_max[0] = matrix[i, j]
                    i_max[0] = i
                    j_max[0] = j

        P = hcl.mutate((lenA+1, lenB+1), lambda i, j: populate_matrix(i, j))

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

        with hcl.Stage("T"):
            curr_i = hcl.local(i_max[0], "curr_i")
            curr_j = hcl.local(j_max[0], "curr_j")
            next_i = hcl.local(0, "next_i")
            next_j = hcl.local(0, "next_j")
            next_i[0], next_j[0] = get_next(action, curr_i[0], curr_j[0])
            tick = hcl.local(0, "tick")

            with hcl.while_(hcl.or_(curr_i[0] != next_i[0],
                                    curr_j[0] != next_j[0])):
                consA[tick[0]], consB[tick[0]] = \
                    align(curr_i, curr_j, next_i, next_j)
                curr_i[0], curr_j[0] = next_i[0], next_j[0]
                next_i[0], next_j[0] = get_next(action, curr_i[0], curr_j[0])
                tick[0] += 1

    def batch_sw(seqAs, seqBs, outAs, outBs):
        hcl.mutate((num,),
                lambda t: smith_waterman(seqAs[t], seqBs[t], outAs[t], outBs[t]),
                "B")

    seqAs = hcl.placeholder((num, lenA), "seqAs", dtype)
    seqBs = hcl.placeholder((num, lenB,), "seqBs", dtype)
    outAs = hcl.placeholder((num, lenA+lenB), "outAs", dtype)
    outBs = hcl.placeholder((num, lenA+lenB), "outBs", dtype)

    # seqAs = hcl.placeholder((num, lenA), "seqAs")
    # seqBs = hcl.placeholder((num, lenB,), "seqBs")
    # outAs = hcl.placeholder((num, lenA+lenB), "outAs")
    # outBs = hcl.placeholder((num, lenA+lenB), "outBs")

    scheme = hcl.create_scheme([seqAs, seqBs, outAs, outBs], batch_sw)
    scheme.downsize([batch_sw.B.matrix, batch_sw.B.action], mtype)
    s = hcl.create_schedule_from_scheme(scheme)
    o, p = s[batch_sw.B].split(batch_sw.B.axis[0], factor=32)
    s[batch_sw.B].pipeline(o)
    # s[batch_sw.B].parallel(p)
    s[batch_sw.B].unroll(p)
    return hcl.build(s, target=target)

###############################################################################
# Test the algorithm with random numbers
_seqA = hcl.asarray(np.random.randint(1, 5, size=(num, lenA)), dtype)
_seqB = hcl.asarray(np.random.randint(1, 5, size=(num, lenB)), dtype)
_consA = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)
_consB = hcl.asarray(np.zeros((num, (lenA + lenB))), dtype)

# _seqA = hcl.asarray(np.random.randint(1, 5, size=(num, lenA)))
# _seqB = hcl.asarray(np.random.randint(1, 5, size=(num, lenB)))
# _consA = hcl.asarray(np.zeros((num, (lenA + lenB))))
# _consB = hcl.asarray(np.zeros((num, (lenA + lenB))))




# f = top()
code = top('sdaccel');
with open('sdaccel_code.cl', 'w') as f:
    f.write(code)

code2 = top('aocl')
with open('smith_aocl.cl', 'w') as fin:
    fin.write(code2)

code3 = top('vhls')
with open('smith_vhls.cl', 'w') as fin:
    fin.write(code3)

assert 1==2


# code3 = top('vhls');
# with open('vhls_code.cl', 'w') as f:
#    f.write(code3)


# code2 = top('merlinc')
# with open('merlinc_code.cl', 'w') as f:
#    f.write(code2)

 

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

# _seqA = hcl.asarray(_seqA_np)
# _seqB = hcl.asarray(_seqB_np)
# _consA = hcl.asarray(np.zeros((num, (lenA + lenB))))
# _consB = hcl.asarray(np.zeros((num, (lenA + lenB))))


f(_seqA, _seqB, _consA, _consB)
_consA_np = _consA.asnumpy()
_consB_np = _consB.asnumpy()

for i in range(0, 256):
    if i < 124:
        assert _consA_np[0][i] == 1
    else:
        assert _consA_np[0][i] == 0
