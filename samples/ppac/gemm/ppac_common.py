import numpy as np

def compact(arr, arrlen, dataBits, wordBits):
    if arr.size < arrlen or wordBits / dataBits < arrlen:
        raise Exception("error in length")
    word = np.zeros(wordBits, dtype=int)
    for l in range(arrlen):
        for b in range(dataBits):
            word[l * dataBits + b] = ((arr[l] & (1 << b))>>b)
    return word

def bvec2x(vec):
    n2c = {0:'0', 1:'1', 2:'2', 3:'3',
           4:'4', 5:'5', 6:'6', 7:'7',
           8:'8', 9:'9',10:'a',11:'b',
           12:'c',13:'d',14:'e',15:'f'}
    base = np.array([1,2,4,8])
    vec_int = vec % 2
    len = vec.size
    if not len % 4 == 0:
        vec_int = np.append(vec_int, np.zeros((4-(len%4))))
        len = len + (4 - len % 4)
    x = ''
    ct = int(len/4)
    for c in range(ct, 0, -1):
        x = x + n2c[(np.dot( vec[4*c-4:4*c], base))]
    return x

def doWrite(rocc_addr):
    s = '\tli a0, '+str(rocc_addr)+'\n'
    return s + '\tROCC_INSTRUCTION_RAW_R_R_R(CUSTOM_X, 0, 11, 10, K_DO_WRITE)\n'

def doRead(rocc_addr):
    s = '\tli a0, '+str(rocc_addr)+'\n'
    return s + '\tROCC_INSTRUCTION_RAW_R_R_R(CUSTOM_X, 10, 0, 10, K_DO_READ)\n'

def doLoad(addr, rocc_addr):
    s = '\tla a1, '+addr+'\n\tli a0, '+str(rocc_addr)+'\n'
    return s + '\tROCC_INSTRUCTION_RAW_R_R_R(CUSTOM_X, 10, 11, 10, K_DO_LOAD)\n'

def doStore(addr, rocc_res_addr):
    s = '\tla a1, ' + addr + '\n\tli a0, ' + str(rocc_res_addr) + '\n'
    return s + '\tROCC_INSTRUCTION_RAW_R_R_R(CUSTOM_X, 10, 11, 10, K_DO_STORE)\n'

def doMVP(funct):
    return '\tROCC_INSTRUCTION_RAW_R_R_R(CUSTOM_X, 10, 11, 10, '+str(funct)+')\n'


