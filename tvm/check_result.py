f = open("result.txt", "r")
for line in f:
    assert("errors" not in line)
f.close()
