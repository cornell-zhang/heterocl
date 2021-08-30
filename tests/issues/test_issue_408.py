import subprocess
import pathlib

def get_stdout():
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path) + "/issue_408_code.py"
    p = subprocess.run(['python', path], stdout=subprocess.PIPE)
    output = p.stdout.decode('utf-8')
    return str(output)

def test_issue_408():

    str = get_stdout()

    assert str == """DRAM: (4, 2) uint64
[[0, 0],
[0, 0],
[0, 0],
[0, 0]]
"""
