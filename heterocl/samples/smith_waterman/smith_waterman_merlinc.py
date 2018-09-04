import heterocl as hcl
from smith_waterman import top
import os

code = top("merlinc")
with open('smith_waterman_kernel.cpp', 'w') as f:
    f.write(code)

# Here we use gcc to evaluate the functionality
os.system('g++ -std=c++11 smith_waterman_host.cpp smith_waterman_kernel.cpp')
os.system('./a.out')
os.system('rm a.out')
