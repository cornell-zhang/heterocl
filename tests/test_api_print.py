# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import subprocess
import heterocl as hcl

unwanted_char = [",", "[", "]", " ", "\n", "\t"]


def get_stdout(filename):
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path) + "/test_api_print_cases/" + filename + ".py"
    p = subprocess.run(["python", path], stdout=subprocess.PIPE)
    output = p.stdout.decode("utf-8")
    return str(output)


def test_print_number():

    output = get_stdout("print_number")

    golden = "5 \n2.500"
    print(output)

    assert golden in str(output)


def test_print_expr():

    outputs = get_stdout("print_expr").split("\n")
    outputs = [x for x in outputs if not (x == "" or "mlir" in x.lower())]
    outputs = [x.strip() for x in outputs]

    assert outputs[10] == "print empty tuple success"

    N = 5
    for i in range(0, N):
        assert outputs[i] == outputs[i + N]


def test_print_tensor_1D():

    outputs = get_stdout("print_tensor_1D").split("\n")

    hcl_print_output = outputs[1]
    np_print_output = outputs[4]

    for c in unwanted_char:
        hcl_print_output = hcl_print_output.replace(c, "")
        np_print_output = np_print_output.replace(c, "")

    assert hcl_print_output == np_print_output


def test_print_tensor_2D():

    outputs = get_stdout("print_tensor_2D").split("\n")

    hcl_print_output = "".join(outputs[1:11])
    np_print_output = "".join(outputs[13:23])

    for c in unwanted_char:
        hcl_print_output = hcl_print_output.replace(c, "")
        np_print_output = np_print_output.replace(c, "")

    assert hcl_print_output == np_print_output


def test_print_tensor_2D_rect():

    outputs = get_stdout("print_tensor_2D_rect").split("\n")

    hcl_print_output = "".join(outputs[1:6])
    np_print_output = "".join(outputs[8:13])

    for c in unwanted_char:
        hcl_print_output = hcl_print_output.replace(c, "")
        np_print_output = np_print_output.replace(c, "")

    assert hcl_print_output == np_print_output


def test_print_tensor_ele():
    outputs = get_stdout("print_tensor_ele").split("\n")
    target_str = outputs[2]
    assert target_str == "here 53"


def test_print_index():
    def kernel():
        with hcl.for_(0, 10) as i:
            hcl.print(i)

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)


def test_print_extra_output():
    output = get_stdout("print_test_extra_char").split("\n")[2]
    # remove all unwanted characters
    for c in unwanted_char:
        output = output.replace(c, "")
    assert len(output) == 46
