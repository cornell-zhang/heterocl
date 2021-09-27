import heterocl as hcl
import numpy as np
import pathlib
import subprocess

def get_stdout(filename):
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path) + "/test_api_assert_cases/" + filename + ".py"
    p = subprocess.run(['python', path], stdout = subprocess.PIPE)
    output = p.stdout.decode('utf-8')
    return str(output)

def test_basic_assert():
    output = get_stdout("basic_assert_tests")
    golden = "in the if statement\ncustomized assert message 1"
    for x in range(7):
        golden += "\nin the first for loop"
    golden += "\nassert message in the second for loop"
    for x in range(7):
        golden += "\nin the first for loop and if statement\nin the first for loop, outside if statement"
    golden += "\nassert message in the second for loop\nassert 0 message  0 number 2\n"

    assert str(output) == golden

def test_memory_assert():
    output = get_stdout("memory_assert_tests")
    golden = "assert message in the if statement 9\n"
    golden += "in if statement\nassert message for loop\n"
    for x in range(2):
        golden += "in if statement\nin for loop\n"
    golden += "assert error, matrix_A[1, 1]: 11 matrix_A[2, 1]: 11 matrix_A[3, 1]: 11\n"
    for x in range(3):
        for x in range(2):
            golden += "in if statement\nin for loop\n"
        golden += "in the while loop\n"
    golden += "assert message end\ncustomized assert message 1\nassert error in if--value of x: 0\n"

    assert str(output) == golden

def test_dsl_def_assert():
    output = get_stdout("dsl_def_assert_tests")
    golden = get_stdout("dsl_def_assert_tests_golden")

    assert str(output) == golden

def test_assert_exception():
    hcl.init()

    A = hcl.placeholder((10,))

    def kernel(A):
        hcl.assert_(5 == 6)
        return hcl.compute(A.shape, lambda x: A[x])

    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    hclA = hcl.asarray(np.zeros(A.shape))
    hclO = hcl.asarray(np.zeros(A.shape))

    try:
        f(hclA, hclO)
    except hcl.debug.AssertError:
        return
    assert False
