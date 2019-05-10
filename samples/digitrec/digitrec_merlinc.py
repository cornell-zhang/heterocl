import os
import heterocl as hcl
from digitrec_main import top

code = top('merlinc')
with open('kernel.cpp', 'w') as f:
    f.write(code)

# Here we use gcc to evaluate the functionality
os.system('g++ -std=c++11 digitrec_host.cpp kernel.cpp')
os.system('./a.out')
os.system('rm a.out')
