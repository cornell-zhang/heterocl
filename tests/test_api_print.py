import heterocl as hcl
import numpy as np
import pathlib
import subprocess

def get_stdout(filename):
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path) + "/test_api_print_cases/" + filename + ".py"
    p = subprocess.run(['python', path], stdout=subprocess.PIPE)
    output = p.stdout.decode('utf-8')
    return str(output)

def test_print_number():

    output = get_stdout("print_number")

    golden = "5\n2.500000\n"

    assert str(output) == golden

def test_print_expr():

    outputs = get_stdout("print_expr").split("\n")

    N = 5
    for i in range(0, N):
        assert outputs[i] == outputs[i+N]

def test_print_tensor_1D():

    outputs = get_stdout("print_tensor_1D").split("\n")

    assert outputs[0] == outputs[1]

def test_print_tensor_2D():

    outputs = get_stdout("print_tensor_2D").split("\n")

    N = 10
    for i in range(0, N):
        assert outputs[i] == outputs[i+N]

def test_print_tensor_2D_rect():

    outputs = get_stdout("print_tensor_2D_rect").split("\n")

    N = 5
    for i in range(0, N):
        assert outputs[i] == outputs[i+N]
