import heterocl as hcl
import numpy as np
from smith_waterman_main import *

f = top("vhls")

fl = open("test.cpp", "w")
fl.write(f)
fl.close()

