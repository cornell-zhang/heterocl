import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)))

def test_digitrec():
    sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/digitrec"))
    from digitrec import digitrec_main

def test_cordic():
    sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/cordic"))
    from cordic import cordic_main

def test_kmeans():
    sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/kmeans"))
    from kmeans import kmeans_main

def test_smith_waterman():
    from smith_waterman import smith_waterman_main

def test_gemm():
    from gemm import gemm_main

def test_fft():
    from fft import fft

def test_lenet():
    from lenet import lenet_main

def test_sobel():
    from sobel import sobel_main 
