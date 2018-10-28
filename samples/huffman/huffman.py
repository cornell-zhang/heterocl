# codeing -*utf-8*-
import numpy as np
import heterocl as hcl

# data type for encoding and channel
DTYPE = hcl.UInt(8)

M = 1024 # training text length
N = 512  # test text length
K = 128  # frequency list size (ASCII representation)

def top(target = None):
    train = hcl.placeholder((M,))
    text  = hcl.placeholder((N,))
    encode  = hcl.placeholder((N,))
    hmatrix = hcl.placeholder((M,7), 'huff')
    heapM = hcl.placeholder((K,2), 'heapM')

    # the encoding function offload to FPGA
    def Huffman(train_text, input_text, huffman, heap, encode):

        with hcl.stage('S') as S:

            # exclude the char with zero-freq
            def find_min(heap, len_):
                min_in = hcl.local(99999, 'heap_init')
                idx_in = hcl.local(0, 'idx')
                entry = hcl.local(0, 'entry')
                with hcl.for_(0, len_) as k:
                    with hcl.if_(hcl.and_(heap[k][1] < min_in[0], heap[k][1] > 0)):
                        min_in[0] = heap[k][1]
                        idx_in[0] = heap[k][0]
                        entry[0] = k

                heap[entry[0]][1] = 0
                return min_in[0], idx_in[0], entry[0]


            def init_huffman(x, y):
                with hcl.if_(hcl.and_(x < 128, y <2)):
                    huffman[x][y] = freq[x][y]
                with hcl.else_():
                    huffman[x][y] = 0

            def update_huffman(n):
                # pop the smallest two elements

                itr = hcl.compute((1,), lambda x: n, 'itr')
                min_[0], index[0], entry[0] = find_min(heap, K)
                _min_[0], _index[0], _entry[0] = find_min(heap, K)

                # merge the node and mark it with the index
                with hcl.if_(_min_[0] != 99999):
                    new_v[0] = _min_[0] + min_[0]
                    new_i[0] = 128 + n

                    # insert to replace index
                    heap[entry[0]][0] = new_i[0]
                    heap[entry[0]][1] = new_v[0]

                    # exchange to evict last position K-t-1
                    with hcl.if_(heap[_entry[0]][0] != K-itr[0]-1):
                        heap[_entry[0]][0] = heap[K-itr[0]-1][0]
                        heap[_entry[0]][1] = heap[K-itr[0]-1][1]
                        heap[K-itr[0]-1][0] = 0
                        heap[K-itr[0]-1][1] = 0

                    # update huffman matrix children and parent
                    # index 2 leftChild 3 rightChild 4 Parent 5 Encode
                    with hcl.if_(huffman[index[0]][2] == 0):
                            huffman[index[0]][2] = -1
                            huffman[index[0]][3] = -1
                            huffman[index[0]][6] = 1
                    with hcl.if_(huffman[_index[0]][2] == 0):
                            huffman[_index[0]][2] = -1
                            huffman[_index[0]][3] = -1
                            huffman[_index[0]][6] = 1

                    huffman[_index[0]][4] = 128 - itr[0]
                    huffman[index[0]][4] = 128 - itr[0]

                    # insert the huffman matrix with new value
                    huffman[128 + itr[0]][0] = 128 + itr[0]
                    huffman[128 + itr[0]][1] = new_v[0]
                    huffman[128 + itr[0]][2] = index[0]
                    huffman[128 + itr[0]][3] = _index[0]
                    huffman[128 + itr[0]][6] = -1

                # begin encoding
                with hcl.else_():
                    root = hcl.local(0, 'root')
                    temp_code = hcl.local(0, 'temp_code')
                    with hcl.for_(0,M) as col:
                        with hcl.if_(huffman[M-1-col][1] != 0):
                            with hcl.if_(huffman[M-1-col][4] == root):
                                huffman[M-1-col][5] = 1
                                temp_code[0] = huffman[M-1-col][2]
                                huffman[temp_code[0]][5] = 0
                                temp_code[0] = huffman[M-1-col][3]
                                huffman[temp_code[0]][5] = 1
                            with hcl.else_():
                                temp_code[0] = huffman[M-1-col][2]
                                with hcl.if_(temp_code[0] != -1):
                                    huffman[temp_code[0]][5] = huffman[M-1-col][5] * 2
                                    temp_code[0] = huffman[M-1-col][3]
                                    huffman[temp_code[0]][5] = huffman[M-1-col][5] * 2 + 1

            def generate_encode(n):
                with hcl.for_(0,N) as i:
                    encode[i] = huffman[input_text[i]][5]

            def init_heap(x, y):
                heap[x][y] = freq[x][y]

            def init_freq(x, y):
                with hcl.if_(y == 0):
                    freq[x][y] = x
                with hcl.else_():
                    freq[x][y] = 0

            def get_frequency(i):
                temp_ord = hcl.compute((1,), lambda x: train_text[i])
                freq[temp_ord[0]][1] = freq[temp_ord[0]][1] + 1
                freq[temp_ord[0]][0] = temp_ord[0]

            def update_freq(freq, x, y, temp_ord):
                with hcl.if_(x == temp_ord[0]):
                    with hcl.if_(y == 1):
                        return freq[x][y] + 1
                    with hcl.else_():
                        return freq[x][y]
                with hcl.else_():
                    return freq[x][y]

            # initialize the frequency array
            freq = hcl.compute((K,2), lambda x, y: 0, 'freq')
            init_freq = hcl.mut_compute((K,2), lambda x, y: init_freq(x, y), 'init_freq')
            get_freq = hcl.mut_compute((M,), lambda n: get_frequency(n), 'get_freq')

            new = hcl.compute(freq.shape, lambda x, y: freq[x][y], 'new')
            init_heap = hcl.mut_compute(heap.shape, lambda x, y: init_heap(x, y), 'init_heap')
            init_huff = hcl.mut_compute(huffman.shape, lambda i, j: init_huffman(i, j), 'init_huff')

            # find first and second find, insert value and update index
            min_ = hcl.compute((1,), lambda x: 0, 'min_')
            _min_ = hcl.compute((1,), lambda x: 0, '_min_')
            index = hcl.compute((1,), lambda x: 0, 'index')
            _index = hcl.compute((1,), lambda x: 0, '_index')
            entry = hcl.compute((1,), lambda x: 0, 'entry')
            _entry = hcl.compute((1,), lambda x: 0, '_entry')
            new_v = hcl.compute((1,), lambda x: 0, 'new_value')
            new_i = hcl.compute((1,), lambda x: 0, 'new_index')

            update_huffman = hcl.mut_compute((K-1,), lambda x: update_huffman(x), 'update_huffman')

            # create encoding from huffman matrix
            X = hcl.mut_compute((1,), lambda x: generate_encode(x), 'X')
        return S

    s = hcl.make_schedule([train, text, hmatrix, heapM, encode], Huffman)
    print hcl.lower(s, [train, text, hmatrix, heapM, encode])

    # make scheme schedule
    return hcl.build(s, [train, text, hmatrix, heapM, encode], target = target)


def read_text(path):
    with open(path, 'r+') as file:
        text = file.read()
        text = text.rstrip()
        return text

# read data from train text
train_text = [ord(c) for c in read_text('sample.txt')][:M]
input_text = [ord(c) for c in read_text('sample.txt')][:N]
encode_array = [0 for i in range(N)]

# build function and test encoding
f = top()

hcl_huffman = hcl.asarray(np.zeros((1024,7)), hcl.Int())
hcl_heap = hcl.asarray(np.zeros((128,2)), hcl.Int())
hcl_train_text = hcl.asarray(np.array(train_text), dtype=hcl.Int())
hcl_input_text = hcl.asarray(np.array(input_text), hcl.Int())
hcl_encode = hcl.asarray(np.array(encode_array), hcl.Int())

f(hcl_train_text, hcl_input_text, hcl_huffman, hcl_heap, hcl_encode)
encode = hcl_encode.asnumpy()
huffman = hcl_huffman.asnumpy()
heap = hcl_heap.asnumpy()

print '\n[INFO] input text'
print np.array(input_text)
print '\n[INFO] encoding result'
print encode
